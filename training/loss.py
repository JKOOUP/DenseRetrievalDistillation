import torch
import typing as tp
import torch.nn.functional as F


from omegaconf import DictConfig

EPS = 1e-9


class ContrastiveCrossBatchLoss(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config: DictConfig = config

    def forward(
        self,
        qids: torch.Tensor,
        cb_qids: torch.Tensor,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        labels: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if labels is None:
            labels = self._build_labels(qids, cb_qids)

        scores: torch.Tensor = (query_embeddings @ doc_embeddings.T) / self.config.temperature
        scores[labels > 0] -= self.config.threshold
        return F.cross_entropy(scores, labels)

    def _build_labels(self, qids: torch.Tensor, cb_qids: torch.Tensor) -> torch.Tensor:
        comparison_mask = (qids.view(-1, 1) == cb_qids.view(1, -1)).type_as(qids)
        labels = comparison_mask / comparison_mask.sum(dim=-1, keepdim=True)
        return labels


class DRContrastiveDistillationLoss(torch.nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config: DictConfig = config
        self.contrastive_cross_batch_loss_fn = ContrastiveCrossBatchLoss(self.config)

    def forward(self, embeddings: tp.Dict[str, tp.Any]) -> torch.Tuple[torch.Tensor]:
        embeddings_loss: torch.Tensor = self._calc_embeddings_loss(embeddings, self.config.embeddings_loss_weight)
        scores_loss: torch.Tensor = self._calc_scores_loss(embeddings, self.config.scores_loss_weight)
        contrastive_cross_batch_loss: torch.Tensor = self._calc_contrative_cross_batch_loss(
            embeddings,
            self.config.contrastive_loss_weight,
            distillation=False
        )
        contrastive_cross_batch__distillation_loss: torch.Tensor = self._calc_contrative_cross_batch_loss(
            embeddings,
            self.config.contrastive_distillation_loss_weight,
            distillation=True,
        )
        return embeddings_loss, scores_loss, contrastive_cross_batch_loss, contrastive_cross_batch__distillation_loss

    def _calc_embeddings_loss(self, embeddings: tp.Dict[str, tp.Any], loss_weight: float) -> torch.Tensor:
        if loss_weight < EPS:
            return torch.tensor(0.0)

        if embeddings["student"]["query_proj"] is not None:
            loss =  F.mse_loss(embeddings["student"]["query_proj"], embeddings["teacher"]["query"])
            loss += F.mse_loss(embeddings["student"]["document_proj"], embeddings["teacher"]["document"])
            return loss_weight * loss

        loss =  F.mse_loss(embeddings["student"]["query"], embeddings["teacher"]["query"])
        loss += F.mse_loss(embeddings["student"]["document"], embeddings["teacher"]["document"])
        return loss_weight * loss

    def _calc_scores_loss(self, embeddings: tp.Dict[str, tp.Any], loss_weight: float) -> torch.Tensor:
        if loss_weight < EPS:
            return torch.tensor(0.0)

        loss: torch.Tensor = self._calc_scores_loss_impl(
            embeddings["student"]["query"],
            embeddings["student"]["cb_document"],
            embeddings["teacher"]["query"],
            embeddings["teacher"]["cb_document"]
        )
        loss += self._calc_scores_loss_impl(
            embeddings["student"]["document"],
            embeddings["student"]["cb_query"],
            embeddings["teacher"]["document"],
            embeddings["teacher"]["cb_query"],
        )

        return loss_weight * loss

    def _calc_scores_loss_impl(
        self,
        student_query_embeddings: torch.Tensor,
        student_doc_embeddings: torch.Tensor,
        teacher_query_embeddings: torch.Tensor,
        teacher_doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        student_scores = student_query_embeddings @ student_doc_embeddings.T
        teacher_scores = teacher_query_embeddings @ teacher_doc_embeddings.T
        return F.mse_loss(student_scores, teacher_scores)

    def _calc_contrative_cross_batch_loss(self, embeddings: tp.Dict[str, tp.Any], loss_weight: float, distillation: bool = False) -> torch.Tensor:
        if loss_weight < EPS:
            return torch.tensor(0.0)

        loss: torch.Tensor = self._calc_contrative_cross_batch_loss_impl(
            embeddings["qids"],
            embeddings["student"]["query"],
            embeddings["cb_qids"],
            embeddings["student"]["cb_document"],
            embeddings["teacher"]["query"],
            embeddings["teacher"]["cb_document"],
            distillation=distillation,
        )
        loss += self._calc_contrative_cross_batch_loss_impl(
            embeddings["docids"],
            embeddings["student"]["document"],
            embeddings["cb_docids"],
            embeddings["student"]["cb_query"],
            embeddings["teacher"]["document"],
            embeddings["teacher"]["cb_query"],
            distillation=distillation,
        )

        return loss_weight * loss

    def _calc_contrative_cross_batch_loss_impl(
        self,
        qids: torch.Tensor,
        query_embeddings: torch.Tensor,
        cb_qids: torch.Tensor,
        cb_doc_embeddings: torch.Tensor,
        teacher_query_embeddings: torch.Tensor,
        teacher_cb_doc_embeddings: torch.Tensor,
        distillation: bool = False,
    ) -> torch.Tensor:
        if distillation:
            teacher_scores: torch.Tensor = teacher_query_embeddings @ teacher_cb_doc_embeddings.T
            distillation_labels: torch.Tensor = torch.softmax(teacher_scores, dim=-1)
            return self.contrastive_cross_batch_loss_fn(None, None, query_embeddings, cb_doc_embeddings, labels=distillation_labels)
        else:
            return self.contrastive_cross_batch_loss_fn(qids, cb_qids, query_embeddings, cb_doc_embeddings)
