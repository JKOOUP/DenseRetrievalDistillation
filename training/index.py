import torch
import typing as tp


class PreAllocatedFlatIndex:
    def __init__(self, index_size: int, emb_dim: int, device: str, dtype: type):
        self.index_size: int = index_size
        self.index_device: str = device
        self.index_dtype: type = dtype

        self.pointer: int = 0
        self.ids: torch.Tensor = -torch.ones(self.index_size, device=device, dtype=torch.long)
        self.index_vectors: torch.Tensor = torch.zeros((self.index_size, emb_dim), device=self.index_device, dtype=self.index_dtype)

    def add_with_ids(self, ids: torch.Tensor, embeddings: torch.Tensor):
        self.ids[self.pointer : self.pointer + ids.shape[0]] = ids.to(self.index_device, torch.long)
        normed_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        self.index_vectors[self.pointer : self.pointer + ids.shape[0]] = normed_embeddings.to(self.index_device, self.index_dtype)
        self.pointer = (self.pointer + ids.shape[0]) % self.index_size

    def search(self, query_embeddings: torch.Tensor, max_top: int, minimum_threshold: float = -1.0, maximum_threshold: float = 1.0):
        normed_query_embeddings: torch.Tensor = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1).to(self.index_device, self.index_dtype)
        pairwise_distances: torch.Tensor = normed_query_embeddings @ self.index_vectors.T
        distances, indices = torch.topk(pairwise_distances, k=max_top, dim=-1)

        docids: tp.List[int] = []
        distances_list: tp.List[float] = []
        for idx in range(query_embeddings.shape[0]):
            threshold_mask = (minimum_threshold < distances[idx]) & (distances[idx] < maximum_threshold)
            distances_list.append(distances[idx, threshold_mask])
            docids.append(self.ids[indices[idx, threshold_mask]])
        return distances_list, docids
