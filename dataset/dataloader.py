import torch
import typing as tp

from functools import partial
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from dataset import DenseRetrievalDataset


def train_collate_fn(
    dataset_items: tp.List[tp.Dict[str, str]],
    config: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tp.Dict[str, tp.Any]:
    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    return {
        "qid": torch.tensor([item["qid"] for item in dataset_items]),
        "docid": torch.tensor([item["docid"] for item in dataset_items]),
        "query": tokenizer([item["query"] for item in dataset_items], max_length=config.query_max_len, **tokenizer_kwargs),
        "document": tokenizer([item["document"] for item in dataset_items], max_length=config.doc_max_len, **tokenizer_kwargs),
    }

def test_collate_fn(
    dataset_items: tp.List[tp.Dict[str, str]],
    config: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tp.Dict[str, tp.Any]:
    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }

    is_all_query: bool = all([item["is_query"] for item in dataset_items])
    is_all_document: bool = all([not item["is_query"] for item in dataset_items])

    if not is_all_document and not is_all_query:
        max_lenght: int = max(config.query_max_len, config.doc_max_len)
    else:
        max_lenght: int = config.query_max_len if is_all_query else config.doc_max_len

    return {
        "zone_id": torch.tensor([item["zone_id"] for item in dataset_items]),
        "zone": tokenizer([item["zone_text"] for item in dataset_items], max_length=max_lenght, **tokenizer_kwargs),
        "is_query": torch.tensor([item["is_query"] for item in dataset_items], dtype=torch.bool),
    }


def build_dataloader(dataset: DenseRetrievalDataset, data_config: DictConfig, test: bool = False) -> DataLoader:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(data_config.tokenizer_path)
    return DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        collate_fn=partial(test_collate_fn if test else train_collate_fn, config=data_config, tokenizer=tokenizer),
        drop_last=data_config.drop_last,
    )
