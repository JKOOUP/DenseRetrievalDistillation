import sys
import torch
import typing as tp

from omegaconf import DictConfig
from transformers import PreTrainedModel, AutoModel, AutoConfig


class BiEncoderAveragePool(torch.nn.Module):
    def __init__(self, model_config: DictConfig):
        super().__init__()

        self.config: DictConfig = model_config
        self.transformers_config: tp.Dict[str, tp.Any] = AutoConfig.from_pretrained(self.config.pretrained_model_name_or_path)
        self.backbone: PreTrainedModel = AutoModel.from_pretrained(self.config.pretrained_model_name_or_path)

        if self.config.add_head:
            self.head: torch.nn.Module = torch.nn.Linear(self.transformers_config.hidden_size, self.config.embedding_size)
            self.head.weight.data.normal_(0, 0.02)
        else:
            assert self.config.embedding_size == self.transformers_config.hidden_size, \
                f"Set `add_head=False`, `embedding_size={self.config.embedding_size}` and `hidden_size={self.transformers_config.hidden_size}`!\n" \
                f"You have to set `embedding_size` == `hidden_size` to train model without head!"
        
        if "checkpoint_path" in self.config:
            checkpoint: tp.Dict[str, torch.Tensor] = torch.load(self.config.checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
                print("Loading weights from lightning checkpoint!", file=sys.stderr)
            
            self.load_state_dict(checkpoint)

    def forward(self, batch: tp.Dict[str, torch.Tensor]) -> torch.Tensor:
        last_hidden_state = self.backbone(**batch).last_hidden_state
        averaged_embeddings = self._average_pool(last_hidden_state, batch["attention_mask"])
        if self.config.add_head:
            averaged_embeddings = self.head(averaged_embeddings)
        output = torch.nn.functional.normalize(averaged_embeddings, p=2, dim=-1)
        return output

    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden: torch.Tensor = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class TeacherModel:
    def __init__(self, config: DictConfig):
        self.config: DictConfig = config
        
        self.model: BiEncoderAveragePool = BiEncoderAveragePool(self.config)
        self.model.eval()

        if self.config.use_cache:
            self.cache_zone_id2idx: tp.Dict[str, tp.Any] = {"query": {}, "document": {}}
            self.cached_embeddings = {
                "query": torch.zeros((0, self.config.embedding_size), device="cpu", dtype=torch.float16),
                "document": torch.zeros((0, self.config.embedding_size), device="cpu", dtype=torch.float16),
            }
    
    @torch.no_grad()
    def __call__(
        self,
        batch: tp.Dict[str, torch.Tensor],
        zone_ids: torch.Tensor,
        is_query: bool,
        avoid_cache: bool = False
    ) -> torch.Tensor:
        if (not self.config.use_cache) or avoid_cache:
            return self.model(batch)

        cache_key: str = "query" if is_query else "document"
        embeddings = self.cached_embeddings[cache_key][
            [self.cache_zone_id2idx[cache_key][zone_id] for zone_id in zone_ids.tolist()]
        ].to(device="cuda", dtype=torch.float16)
        return embeddings

    def _store_cache(
            self,
            qids: torch.Tensor,
            docids: torch.Tensor,
            query_embeddings: torch.Tensor,
            doc_embeddings: torch.Tensor
        ) -> torch.Tensor:
        for idx, qid in enumerate(qids.tolist()):
            self.cache_zone_id2idx["query"][qid] = idx
        for idx, docid in enumerate(docids.tolist()):
            self.cache_zone_id2idx["document"][docid] = idx
        
        self.cached_embeddings["query"] = query_embeddings.to(device="cpu", dtype=torch.float16)
        self.cached_embeddings["document"] = doc_embeddings.to(device="cpu", dtype=torch.float16)


class StudentModel(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config: DictConfig = config
        self.model: BiEncoderAveragePool = BiEncoderAveragePool(self.config)

        if "return_proj_to" in self.config:
            self.proj: torch.nn.Linear = torch.nn.Linear(self.config.embedding_size, self.config.return_proj_to)
            self.proj.weight.data.normal_(0, 0.02)

    def forward(self, batch):
        model_output: torch.Tensor = self.model(batch)
        if "return_proj_to" in self.config:
            return self.model(batch), torch.nn.functional.normalize(self.proj(model_output), p=2, dim=-1)
        return model_output
