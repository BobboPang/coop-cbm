"""
Graph-COL: 基于概念分组的图结构正交损失

利用 CUB 数据集中 312 个属性的 28 个语义分组（如 wing_color, bill_shape 等），
在训练端改进概念质量：
1. 组内聚合 (ConceptGroupGNN)：同组概念共享上下文信息，提升预测一致性
2. 组级正交损失 (GraphCOLLoss)：同组特征聚类、不同组特征正交
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util.config import CUB_DATA_DIR, N_ATTRIBUTES


def build_concept_groups(n_attributes=N_ATTRIBUTES):
    """从 CUB attributes.txt 构建概念分组结构。

    按属性名前缀（前10字符）将 312 个属性分为 28 组，
    如 has_wing_color, has_bill_shape 等各为一组。

    Returns:
        concept_groups: dict, group_id -> list of attribute indices (0-based)
        group_to_range: dict, group_id -> (start_idx, end_idx) for slicing
        n_groups: int, 总组数 (28)
    """
    attr_file = os.path.join(CUB_DATA_DIR, 'attributes', 'attributes.txt')
    if not os.path.exists(attr_file):
        # 回退：均匀分组
        n_groups = 28
        group_size = n_attributes // n_groups
        concept_groups = {}
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else n_attributes
            concept_groups[g] = list(range(start, end))
        return concept_groups, n_groups

    # 按属性名前缀分组
    attr_names = []
    with open(attr_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            attr_names.append((int(parts[0]) - 1, parts[1]))

    concept_groups = {}
    group_id = 0
    prev_prefix = attr_names[0][1][:10]
    concept_groups[group_id] = [attr_names[0][0]]

    for idx, name in attr_names[1:]:
        prefix = name[:10]
        if prefix != prev_prefix:
            group_id += 1
            prev_prefix = prefix
            concept_groups[group_id] = [idx]
        else:
            concept_groups[group_id].append(idx)

    n_groups = group_id + 1
    return concept_groups, n_groups


class ConceptGroupGNN(nn.Module):
    """组内概念聚合模块：利用注意力机制在同组概念间传递上下文。

    每个概念的最终预测 = 原始 logit + alpha * 组内加权均值，
    使同组概念（如 wing_color_blue 和 wing_color_black）共享上下文。
    """

    def __init__(self, n_attributes, concept_groups, n_groups, alpha=0.3):
        super().__init__()
        self.n_attributes = n_attributes
        self.concept_groups = concept_groups
        self.n_groups = n_groups
        # 每组一个可学习的混合权重，控制个体与组上下文的平衡
        self.group_alpha = nn.Parameter(torch.full((n_groups,), alpha))
        # 组内注意力：每组一个小型注意力网络
        self.group_attn = nn.ModuleList()
        for g in range(n_groups):
            group_size = len(concept_groups[g])
            # 简单的双层 MLP 注意力: logit -> attention weight
            self.group_attn.append(nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
            ))

    def forward(self, concept_logits):
        """
        Args:
            concept_logits: (N, 312) 各概念头的原始 logits
        Returns:
            enhanced_logits: (N, 312) 融合组内上下文后的 logits
        """
        N = concept_logits.shape[0]
        enhanced = concept_logits.clone()

        for g in range(self.n_groups):
            attr_indices = self.concept_groups[g]
            group_logits = concept_logits[:, attr_indices]  # (N, group_size)

            # 计算组内注意力权重
            attn_weights = []
            for i in range(len(attr_indices)):
                w = self.group_attn[g](group_logits[:, i:i+1])  # (N, 1)
                attn_weights.append(w)
            attn_weights = torch.cat(attn_weights, dim=1)  # (N, group_size)
            attn_weights = F.softmax(attn_weights, dim=1)

            # 组内加权均值作为上下文
            group_context = (attn_weights * group_logits).sum(dim=1, keepdim=True)  # (N, 1)

            # 融合: individual + alpha * group_context
            alpha = torch.sigmoid(self.group_alpha[g])  # 约束在 [0,1]
            for i, attr_idx in enumerate(attr_indices):
                enhanced[:, attr_idx] = concept_logits[:, attr_idx] + alpha * group_context.squeeze(1)

        return enhanced


class GraphCOLLoss(nn.Module):
    """基于概念分组的图结构正交损失。

    与原始 COL（以类别标签 Y 区分正负对）不同，
    Graph-COL 以概念组 G 区分正负对：
    - 组内一致性：同组概念对同类样本应产生相似激活模式
    - 组间正交性：不同组的激活模式应正交（语义独立）

    两级损失：
    1. 组级 COL：以组特征为单位的正交约束（同类→相似，异类→正交）
    2. 组间正交：不同组特征间应低相关（语义维度互不干扰）
    """

    def __init__(self, gamma=0.5, concept_groups=None, n_groups=28):
        super().__init__()
        self.gamma = gamma
        self.concept_groups = concept_groups
        self.n_groups = n_groups

    def forward(self, concept_preds, labels):
        """
        Args:
            concept_preds: (N, 312) sigmoid 后的概念预测
            labels: (N,) 类别标签
        Returns:
            loss: Graph-COL 损失值
        """
        device = concept_preds.device
        N = concept_preds.shape[0]

        # 构建组级特征：每组概念的均值激活
        group_features = []
        for g in range(self.n_groups):
            attr_indices = self.concept_groups[g]
            group_feat = concept_preds[:, attr_indices].mean(dim=1)  # (N,)
            group_features.append(group_feat)
        group_features = torch.stack(group_features, dim=1)  # (N, n_groups)

        # L2 归一化组特征
        group_features = F.normalize(group_features, p=2, dim=-1)

        # 类别标签掩码
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(N, device=device).bool()
        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()

        # 组级 COL：同类样本的组特征应相似，异类应正交
        dot_prod = torch.matmul(group_features, group_features.t())  # (N, N)
        pos_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)
        group_col_loss = (1.0 - pos_mean) + self.gamma * neg_mean

        # 组间正交损失：不同组的激活模式应低相关
        # 计算 (n_groups, n_groups) 的组间相关矩阵
        inter_group_loss = 0.0
        for g1 in range(self.n_groups):
            for g2 in range(g1 + 1, self.n_groups):
                corr = F.cosine_similarity(
                    group_features[:, g1].unsqueeze(0),
                    group_features[:, g2].unsqueeze(0),
                    dim=1
                ).abs().mean()
                inter_group_loss += corr
        inter_group_loss = inter_group_loss / (self.n_groups * (self.n_groups - 1) / 2)

        # 组内一致性损失：同组概念对同类样本应高度相关
        intra_consistency = 0.0
        for g in range(self.n_groups):
            attr_indices = self.concept_groups[g]
            if len(attr_indices) < 2:
                continue
            group_preds = concept_preds[:, attr_indices]  # (N, group_size)
            # 计算组内概念间的平均相关系数
            group_preds_norm = F.normalize(group_preds - group_preds.mean(dim=0, keepdim=True), p=2, dim=0)
            corr_matrix = torch.matmul(group_preds_norm.t(), group_preds_norm) / N
            # 只取非对角线元素（概念间相关）
            n_concepts = len(attr_indices)
            mask_intra = ~torch.eye(n_concepts, device=device).bool()
            avg_corr = corr_matrix[mask_intra].abs().mean()
            # 同类样本中同组概念应正相关 → 最大化相关
            intra_consistency += (1.0 - avg_corr)
        intra_consistency = intra_consistency / self.n_groups

        # 总损失：组级COL + 组间正交 + 组内一致性
        total_loss = group_col_loss + self.gamma * inter_group_loss + 0.1 * intra_consistency
        return total_loss