import tqdm
import torch
import numpy as np
import typing as tp

from collections import defaultdict

from training import PreAllocatedFlatIndex


def calc_dr_recall(
    index: PreAllocatedFlatIndex,
    queries_emb: torch.Tensor, 
    query_ids: torch.Tensor,
    queries_rel_docids: tp.Dict[int, tp.List[int]],
    top_list: tp.List[int],
    batch_size: int = 128,
) -> tp.List[float]:
    recalls: tp.Dict[int, tp.List[int]] = defaultdict(list)
    for idx in tqdm.tqdm(range(0, len(queries_emb), batch_size)):
        batch_queries_emb: torch.Tensor = queries_emb[idx : idx + batch_size]
        batch_query_ids: torch.Tensor = query_ids[idx : idx + batch_size]

        _, batch_pred_docids = index.search(batch_queries_emb, top_list[-1] + 1)
        for query_id, pred_docids in zip(batch_query_ids.cpu().numpy(), batch_pred_docids):
            true_docids: tp.List[int] = queries_rel_docids[query_id]
            if not true_docids:
                continue

            if isinstance(pred_docids, torch.Tensor):
                pred_docids = pred_docids.detach().cpu().numpy()

            num_true_positives: int = 0
            for curr_top, pred_docid in enumerate(pred_docids):
                if curr_top in top_list:
                    recall_denominator: int = curr_top if curr_top < len(true_docids) else len(true_docids)
                    recalls[curr_top].append(min(1, num_true_positives / recall_denominator))
                num_true_positives += int(pred_docid in true_docids)

    result: tp.List[float] = []
    for top in sorted(recalls.keys()):
        result.append(np.mean(recalls[top]))
    return result
