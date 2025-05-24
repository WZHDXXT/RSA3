# utils/metrics.py

def recall_at_k(preds, targets, k=10):
    """
    计算 Recall@K
    :param preds: List[List[item_id]]，每个用户的推荐列表
    :param targets: List[item_id]，每个用户的真实点击商品
    :param k: int，K值
    :return: float, 平均 Recall@K
    """
    hit_count = 0
    for pred, target in zip(preds, targets):
        if target in pred[:k]:
            hit_count += 1
    return hit_count / len(targets)


def precision_at_k(preds, targets, k=10):
    """
    计算 Precision@K
    """
    precision_scores = []
    for pred, target in zip(preds, targets):
        precision_scores.append(int(target in pred[:k]) / k)
    return sum(precision_scores) / len(precision_scores)


def ndcg_at_k(preds, targets, k=10):
    """
    计算 NDCG@K
    """
    def dcg(rel):
        return sum([(2 ** r - 1) / log2(i + 2) for i, r in enumerate(rel)])

    from math import log2
    ndcg_scores = []
    for pred, target in zip(preds, targets):
        rel = [1 if item == target else 0 for item in pred[:k]]
        ideal_rel = sorted(rel, reverse=True)
        dcg_val = dcg(rel)
        idcg_val = dcg(ideal_rel)
        ndcg_scores.append(dcg_val / idcg_val if idcg_val > 0 else 0.0)
    return sum(ndcg_scores) / len(ndcg_scores)
