import gc
import re
import math
import torch
import typing as tp

from torch.optim import AdamW
from omegaconf import DictConfig
from lightning import LightningModule
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from dataset import (
    DenseRetrievalDataset,
    TestDenseRetrievalDataset,
    build_dataloader
)
from training import (
    DRContrastiveDistillationLoss,
    PreAllocatedFlatIndex,
    TeacherModel,
    StudentModel,
    calc_dr_recall,
)


class LightningModel(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        self.teacher_model: TeacherModel = TeacherModel(self.config.model.teacher)

        if (self.config.model.teacher.embedding_size != self.config.model.student.embedding_size) and \
           (self.config.loss.embeddings_loss_weight > 1e-9):
            self.config.model.student["return_proj_to"] = self.config.model.teacher.embedding_size
        
        self.student_model: StudentModel = StudentModel(self.config.model.student)
        self._unfreeze_parameters(self.config.model.student.unfreeze_pattern)

        self.criterion: DRContrastiveDistillationLoss = DRContrastiveDistillationLoss(self.config.loss)


    def forward(self, batch: tp.Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.student_model(batch)

    def on_train_start(self) -> None:
        self.teacher_model.model.to(device=next(iter(self.student_model.parameters())).device)

    def on_validation_start(self) -> None:
        self.validation_outputs: tp.List[tp.Dict[str, torch.Tensor]] = []

    def on_test_start(self):
        self.teacher_model.model.to(device=next(iter(self.student_model.parameters())).device)

    def training_step(self, batch: tp.Dict[str, tp.Any]) -> tp.Dict[str, torch.Tensor]:
        if "return_proj_to" in self.config.model.student:
            student_query_embeddings, student_query_embeddings_proj = self(batch["query"])
            student_doc_embeddings, student_doc_embeddings_proj = self(batch["document"])
        else:
            student_query_embeddings, student_doc_embeddings = self(batch["query"]), self(batch["document"])
        
        teacher_query_embeddings  = self.teacher_model(batch["query"], batch["qid"], is_query=True)
        teacher_doc_embeddings = self.teacher_model(batch["document"], batch["docid"], is_query=False)

        cb_qids, cb_student_query_embeddings, cb_teacher_query_embeddings = \
            self._gather_cross_batch_zoneids_and_embeddings(batch["qid"], student_query_embeddings, teacher_query_embeddings, sync_grads=True)
        cb_docids, cb_student_doc_embeddings, cb_teacher_doc_embeddings = \
            self._gather_cross_batch_zoneids_and_embeddings(batch["docid"], student_doc_embeddings, teacher_doc_embeddings, sync_grads=True)

        embeddings_dict = {
            "qids": batch["qid"], "cb_qids": cb_qids, "docids": batch["docid"], "cb_docids": cb_docids,
            "student": {
                "query": student_query_embeddings,
                "query_proj": student_query_embeddings_proj if "return_proj_to" in self.config.model.student else None,
                "document": student_doc_embeddings,
                "document_proj": student_doc_embeddings_proj if "return_proj_to" in self.config.model.student else None,
                "cb_query": cb_student_query_embeddings,
                "cb_document": cb_student_doc_embeddings,
            },
            "teacher": {
                "query": teacher_query_embeddings,
                "document": teacher_doc_embeddings,
                "cb_query": cb_teacher_query_embeddings,
                "cb_document": cb_teacher_doc_embeddings,
            },
        }

        embeddings_loss, scores_loss, retrieval_loss, retrieval_distill_loss = self.criterion(embeddings_dict)
        loss = embeddings_loss + scores_loss + retrieval_loss + retrieval_distill_loss

        self._log_train_step_metrics(
            embeddings_loss.item(), scores_loss.item(), retrieval_loss.item(), retrieval_distill_loss.item(), loss.item()
        )
        return {"loss": loss}

    def validation_step(self, batch: tp.Dict[str, tp.Any]) -> tp.Dict[str, torch.Tensor]:
        student_embeddings: torch.Tensor = self(batch["zone"])[0] if "return_proj_to" in self.student_model.config else self(batch["zone"])
        self.validation_outputs.append({"zone_ids": batch["zone_id"], "student_embeddings": student_embeddings, "is_query": batch["is_query"]})

    def test_step(self, batch: tp.Dict[str, tp.Any]) -> tp.Dict[str, torch.Tensor]:
        teacher_embeddings: torch.Tensor = self.teacher_model(
            None, batch["zone"], is_query=False, avoid_cache=True
        ).to(device="cpu", dtype=torch.float16)
        return {"zone_ids": batch["zone_id"], "teacher_embeddings": teacher_embeddings, "is_query": batch["is_query"]}

    def on_validation_epoch_end(self) -> None:
        for dataloader_idx, (dataset_name, val_dataset) in enumerate(self.val_datasets.items()):
            curr_outputs: tp.List[tp.Dict[str, torch.Tensor]] = self.validation_outputs
            if len(self.val_datasets) > 1:
                curr_outputs = curr_outputs[dataloader_idx]

            qids, docids, query_embeddings, doc_embeddings = self._prepare_validation_outputs(curr_outputs)
            self._calc_and_log_metrics(qids, docids, query_embeddings, doc_embeddings, dataset_name, val_dataset._positive_docids_mapping)
            torch.distributed.barrier()
        self.validation_outputs = []

    def test_epoch_end(self, outputs):
        qids, docids, query_embeddings, doc_embeddings = self._prepare_teacher_embeddings_precalc_outputs(outputs)
        self.teacher_model.store_cache(qids, docids, query_embeddings, doc_embeddings)

    def setup(self, stage: str):
        self.train_dataset = DenseRetrievalDataset(
            self.config.data.train_dataset.path,
            self.config.data.train_dataset.positive_thr,
        )

        self.epoch_len = math.ceil(len(self.train_dataset) / self.config.training.n_gpus)
        self.epoch_len = math.ceil(self.epoch_len / self.config.data.train_dataset.batch_size)

        self.val_datasets: tp.Dict[str, TestDenseRetrievalDataset] = {}
        for val_dataset_params in self.config.data.val_datasets:
            self.val_datasets[val_dataset_params.dataset_name] = TestDenseRetrievalDataset(
                val_dataset_params.path,
                val_dataset_params.positive_thr,
            )

        if self.config.training.use_teacher_embeddings_cache:
            self.test_dataset = TestDenseRetrievalDataset(
                self.config.data.train_dataset.path,
                self.config.data.train_dataset.positive_thr
            )

    def train_dataloader(self) -> DataLoader:
        return build_dataloader(self.train_dataset, self.config.data.train_dataset)

    def val_dataloader(self) -> tp.List[DataLoader]:
        val_dataloaders: tp.List[DataLoader] = []
        for val_dataset, val_dataset_config in zip(self.val_datasets.values(), self.config.data.val_datasets):
             val_dataloaders.append(build_dataloader(val_dataset, val_dataset_config, test=True))
        return val_dataloaders

    def test_dataloader(self) -> DataLoader:
        return build_dataloader(self.train_dataset, self.config.data.train_dataset, test=True)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.weight", "word_embeddings"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": self.config.training.wd},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6, lr=1e-5)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.training.lr,
            epochs=self.config.training.epochs,
            steps_per_epoch=math.ceil(self.epoch_len / self.config.training.accumulate_grad_batches),
            cycle_momentum=self.config.training.cycle_momentum,
            pct_start=self.config.training.pct_start,
            anneal_strategy=self.config.training.anneal_strategy,
            div_factor=self.config.training.div_factor
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": self.config.training.accumulate_grad_batches
        }
        return [optimizer], [scheduler_dict]

    def _prepare_validation_outputs(
        self,
        outputs: tp.List[tp.Dict[str, torch.Tensor]]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gathered_outputs = self._gather_validation_step_outputs_batched(outputs)
        qids = gathered_outputs["zone_ids"][gathered_outputs["is_query"]]
        docids = gathered_outputs["zone_ids"][~gathered_outputs["is_query"]]
        query_embeddings = gathered_outputs["student_embeddings"][gathered_outputs["is_query"]]
        doc_embeddings = gathered_outputs["student_embeddings"][~gathered_outputs["is_query"]]
        return qids, docids, query_embeddings, doc_embeddings

    def _prepare_teacher_embeddings_precalc_outputs(
        self,
        outputs: tp.List[tp.Dict[str, torch.Tensor]],
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gathered_outputs = self._gather_validation_step_outputs_batched(outputs)
        qids = gathered_outputs["zone_ids"][gathered_outputs["is_query"]]
        docids = gathered_outputs["zone_ids"][~gathered_outputs["is_query"]]
        query_embeddings = gathered_outputs["teacher_embeddings"][gathered_outputs["is_query"]]
        doc_embeddings = gathered_outputs["teacher_embeddings"][~gathered_outputs["is_query"]]
        return qids, docids, query_embeddings, doc_embeddings

    def _log_train_step_metrics(
        self,
        embeddings_loss: float,
        scores_loss: float, 
        retrieval_loss: float, 
        retrieval_distill_loss: float,
        loss: float
    ) -> None:
        self.log_dict({
            "train/embeddings_loss": embeddings_loss,
            "train/scores_loss": scores_loss,
            "train/retrieval_loss": retrieval_loss,
            "train/retrieval_distill_loss": retrieval_distill_loss,
            "train/loss": loss,
        }, on_step=True, on_epoch=False, sync_dist=True)

    def _calc_and_log_metrics(
        self,
        qids: torch.Tensor,
        docids: torch.Tensor,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        dataset_name: str,
        positive_docids_mapping: tp.Dict[str, tp.List[int]]
    ) -> None:
        self._rebuild_index(docids, doc_embeddings)

        recalls: tp.List[float] = calc_dr_recall(self.index, query_embeddings, qids, positive_docids_mapping, self.config.training.top_list)
        for idx, top in enumerate(self.config.training.top_list):
            self.log(f"{dataset_name}/recall@{top}", recalls[idx], sync_dist=True, reduce_fx=max)

        self.index = None

    def _rebuild_index(self, docids: torch.Tensor, doc_embeddings: torch.Tensor) -> None:
        gc.collect()
        torch.cuda.empty_cache()
        self.index: PreAllocatedFlatIndex = PreAllocatedFlatIndex(len(doc_embeddings), doc_embeddings.shape[-1], doc_embeddings.device, doc_embeddings.dtype)
        self.index.add_with_ids(docids, doc_embeddings)

    def _gather_validation_step_outputs_batched(self, outputs: tp.Dict[str, torch.Tensor], batch_size: int = 5 * 10 ** 5) -> tp.Dict[str, torch.Tensor]:
        concatenated_outputs: tp.Dict[str, torch.Tensor] = {}
        for key in outputs[0]:
            concatenated_outputs[key] = torch.cat([step_output[key] for step_output in outputs], dim=0)

        gathered_batches: tp.List[tp.Dict[str, torch.Tensor]] = []
        for batch_start in range(0, len(concatenated_outputs["zone_ids"]), batch_size):
            local_batch: tp.Dict[str, torch.Tensor] = {
                key: value[batch_start : batch_start + batch_size].cuda() for key, value in concatenated_outputs.items()
            }
            gathered_batch: tp.Dict[str, torch.Tensor] = self.all_gather(local_batch)
            gathered_batches.append({key: value.cpu() for key, value in gathered_batch.items()})

        gathered_outputs: tp.Dict[str, torch.Tensor] = {key: [] for key in concatenated_outputs}
        for rank in range(torch.distributed.get_world_size()):
            for key in gathered_outputs:
                gathered_outputs[key].append(torch.cat([gathered_batch[key][rank] for gathered_batch in gathered_batches], dim=0))
        
        for key, value in gathered_outputs.items():
            gathered_outputs[key] = torch.cat(value, dim=0)
    
        gathered_outputs["is_query"] = gathered_outputs["is_query"].to(dtype=torch.bool)
        return gathered_outputs

    def _gather_cross_batch_zoneids_and_embeddings(self, zone_ids, student_zone_embeddings, teacher_zone_embeddings, sync_grads=True):
        gathering_data = (zone_ids, student_zone_embeddings, teacher_zone_embeddings)
        cb_zone_ids, cb_student_zone_embeddings, cb_teacher_zone_embeddings = self.all_gather(gathering_data, sync_grads=sync_grads)
        world_size, batch_size, student_embedding_dim = cb_student_zone_embeddings.shape
        _, _, teacher_embedding_dim = cb_teacher_zone_embeddings.shape
        return (cb_zone_ids.view(world_size * batch_size), 
                cb_student_zone_embeddings.view(world_size * batch_size, student_embedding_dim), 
                cb_teacher_zone_embeddings.view(world_size * batch_size, teacher_embedding_dim))

    def _unfreeze_parameters(self, pattern):
        for name, parameter in self.student_model.model.named_parameters():
            parameter.requires_grad = False
            if re.search(pattern, name):
                parameter.requires_grad = True
