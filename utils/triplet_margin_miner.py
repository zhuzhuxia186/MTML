import torch

from utils import loss_and_miner_utils as lmu
from .base_miner import BaseMiner


class TripletMarginMiner(BaseMiner):
    """
    Returns triplets that violate the margin
    Args:
        margin
        type_of_triplets: options are "all", "hard", or "semihard".
                "all" means all triplets that violate the margin
                "hard" is a subset of "all", but the negative is closer to the anchor than the positive
                "semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(self, margin=0.05, type_of_triplets="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.type_of_triplets = type_of_triplets
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(labels, ref_labels)
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist

        self.set_stats(ap_dist, an_dist, triplet_margin)

        # 使用布尔索引进行筛选
        anchor_labels = labels[anchor_idx]
        positive_labels = ref_labels[positive_idx]
        negative_labels = ref_labels[negative_idx]

        # 筛选条件
        condition_hand = (anchor_labels < 2) & (positive_labels == anchor_labels) & (triplet_margin <= self.margin)
        condition_foot_tongue = (anchor_labels >= 2) & (positive_labels == anchor_labels) & (
                negative_labels != anchor_labels) & (triplet_margin <= self.margin)

        # 合并条件
        valid_mask = condition_hand | condition_foot_tongue

        # 应用筛选
        new_anchor_idx = anchor_idx[valid_mask]
        new_positive_idx = positive_idx[valid_mask]
        new_negative_idx = negative_idx[valid_mask]

        # 如果没有找到任何有效的三元组，返回空张量
        if len(new_anchor_idx) == 0:
            empty = torch.tensor([], device=embeddings.device, dtype=torch.long)
            return empty, empty, empty

        return new_anchor_idx, new_positive_idx, new_negative_idx

    def set_stats(self, ap_dist, an_dist, triplet_margin):
        if self.collect_stats:
            with torch.no_grad():
                self.pos_pair_dist = torch.mean(ap_dist).item()
                self.neg_pair_dist = torch.mean(an_dist).item()
                self.avg_triplet_margin = torch.mean(triplet_margin).item()
