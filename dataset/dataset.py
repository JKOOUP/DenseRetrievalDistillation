import sys
import hashlib
import numpy as np
import typing as tp

from torch.utils.data import Dataset


class DenseRetrievalDataset(Dataset):
    def __init__(self, data_path: str, positive_thr: int):
        super().__init__()
        self._positive_thr: int = positive_thr

        dataset_lines: tp.Dict[str, tp.Union[tp.List[str], tp.List[int]]] = self._load_dataset_lines(data_path)
        self._qid2query_text, self._docid2doc_text, self._positive_docids_mapping = self._process_dataset_lines(dataset_lines)
        self._positive_pairs: tp.List[tp.Tuple[int, int]] = self._build_positive_pairs()

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Union[int, str]]:
        return {
            "qid": self._positive_pairs[idx][0],
            "docid": self._positive_pairs[idx][1],
            "query": self._qid2query_text[self._positive_pairs[idx][0]],
            "document": self._docid2doc_text[self._positive_pairs[idx][1]],
        }

    def __len__(self) -> int:
        return len(self._positive_pairs)

    def _build_positive_pairs(self) -> tp.List[tp.Tuple[int, int]]:
        positive_pairs: tp.List[tp.Tuple[int, int]] = []
        for qid, positive_docids in self._positive_docids_mapping.items():
            for docid in positive_docids:
                positive_pairs.append((qid, docid))
        return positive_pairs

    def _process_dataset_lines(
        self,
        dataset_lines: tp.Dict[str, tp.Union[tp.List[str], tp.List[int]]],
    ) -> tp.Tuple[tp.Dict[int, str], tp.Dict[int, str], tp.Dict[int, tp.List[int]]]:
        qids, qid2query_text = self._unify_text_zone(dataset_lines["query"])
        docids, docid2doc_text = self._unify_text_zone(dataset_lines["document"])
        positive_docids_mapping: tp.Dict[int, tp.List[int]] = {qid: [] for qid in qids}
        for qid, docid, label in zip(qids, docids, dataset_lines["label"]):
            if label >= self._positive_thr:
                positive_docids_mapping[qid].append(docid)
        return qid2query_text, docid2doc_text, positive_docids_mapping

    def _unify_text_zone(self, zone_texts: tp.List[str]) -> tp.Tuple[tp.List[int], tp.Dict[int, str]]:
        zone_hashes: tp.List[str] = [hashlib.md5(zone_text.encode()).hexdigest() for zone_text in zone_texts]
        _, zone_indices = np.unique(zone_hashes, return_inverse=True)
        return zone_indices, {idx: zone_text for idx, zone_text in zip(zone_indices, zone_texts)}

    def _load_dataset_lines(self, data_path: str, skip_empty_lines: bool = True) -> tp.Dict[str, tp.Union[tp.List[str], tp.List[int]]]:
        num_lines_with_empty_zones: int = 0

        dataset_lines: tp.Dict[str, tp.Union[tp.List[str], tp.List[int]]] = {
            "query": [],
            "document": [],
            "label": [],
        }
        with open(data_path, "r") as dataset_file:
            for idx, line in enumerate(dataset_file):
                try:
                    query, text, label = line.rstrip("\n").split("\t")
                    label = int(label)
                except Exception as err:
                    print(
                        f"Line `{idx}` of dataset cannot be loaded and will be skipped!\n"
                        f"Line: `{line}`\nError: `{err}`",
                        file=sys.stderr
                    )
                    continue

                if skip_empty_lines and ((len(query.strip()) == 0) or (len(text.strip()) == 0)):
                    num_lines_with_empty_zones += 1
                    continue

                dataset_lines["query"].append(query.strip().lower())
                dataset_lines["document"].append(text.strip().lower())
                dataset_lines["label"].append(label)
        
        if num_lines_with_empty_zones:
            print(f"Skipped {num_lines_with_empty_zones} lines with empty zones!", file=sys.stderr)
        return dataset_lines


class TestDenseRetrievalDataset(DenseRetrievalDataset):
    def __init__(self, data_path: str, positive_thr: int):
        super(DenseRetrievalDataset, self).__init__()
        self._positive_thr: int = positive_thr

        dataset_lines: tp.Dict[str, tp.Union[tp.List[str], tp.List[int]]] = self._load_dataset_lines(data_path, skip_empty_lines=False)
        self._qid2query_text, self._docid2doc_text, self._positive_docids_mapping = self._process_dataset_lines(dataset_lines)

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Union[int, str]]:
        if idx < len(self._qid2query_text):
            return {"zone_id": idx, "zone_text": self._qid2query_text[idx], "is_query": True}
        else:
            idx -= len(self._qid2query_text)
            return {"zone_id": idx, "zone_text": self._docid2doc_text[idx], "is_query": False}

    def __len__(self) -> int:
        return len(self._qid2query_text) + len(self._docid2doc_text)
