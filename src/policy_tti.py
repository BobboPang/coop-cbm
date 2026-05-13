"""
Policy-TTI: 可学习的测试时干预策略

将 TTI 建模为主动学习问题：
- 状态：当前已知概念的置信度向量
- 动作：选择查询哪个概念组
- 奖励：查询后分类熵的降低量

训练一个轻量策略网络，在最少查询次数下最大化分类准确率，
替代启发式熵方法，使 TTI 曲线前期斜率更陡。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy


class PolicyNetwork(nn.Module):
    """干预策略网络：根据概念置信度状态选择下一个干预组。

    输入：312 维概念 sigmoid 预测（当前置信度）
    输出：28 维组选择概率（softmax 后的动作分布）
    """

    def __init__(self, n_attributes=312, n_groups=28, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_attributes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_groups),
        )

    def forward(self, concept_confidence):
        """
        Args:
            concept_confidence: (N, 312) sigmoid 后的概念预测
        Returns:
            action_logits: (N, 28) 组选择 logits（未 softmax）
        """
        return self.net(concept_confidence)

    def select_action(self, concept_confidence, intervened_groups=None):
        """选择下一个干预组，排除已干预的组。

        Args:
            concept_confidence: (312,) 单样本概念置信度
            intervened_groups: set, 已干预的组 ID
        Returns:
            group_id: int, 选中的组
            log_prob: float, 该动作的对数概率（用于 REINFORCE）
        """
        x = concept_confidence.unsqueeze(0)  # (1, 312)
        logits = self.net(x).squeeze(0)  # (28,)

        # 排除已干预的组
        if intervened_groups is not None:
            for g in intervened_groups:
                logits[g] = -1e9

        log_probs = F.log_softmax(logits, dim=0)
        probs = F.softmax(logits, dim=0)

        # 从策略分布中采样
        group_id = torch.multinomial(probs, 1).item()
        log_prob = log_probs[group_id]

        return group_id, log_prob


class PolicyTTITrainer:
    """使用 REINFORCE 训练干预策略网络。

    奖励 = 干预后分类熵的降低量，
    即策略应学会在最少干预次数下最大程度降低分类不确定性。
    """

    def __init__(self, policy_net, n_attributes=312, n_groups=28,
                 lr=1e-3, gamma_discount=0.9):
        self.policy = policy_net
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        self.n_attributes = n_attributes
        self.n_groups = n_groups
        self.gamma_discount = gamma_discount

    def compute_entropy(self, class_probs):
        """计算分类分布的熵。"""
        if isinstance(class_probs, np.ndarray):
            return entropy(class_probs)
        return -torch.sum(class_probs * torch.log(class_probs + 1e-10)).item()

    def train_episode(self, concept_preds_sigmoid, attr_labels, model2,
                      concept_groups, ptl_5, ptl_95, max_steps=10,
                      use_relu=False, use_sigmoid=False, device='cuda'):
        """训练一个 episode：从初始状态逐步干预，收集奖励并更新策略。

        Args:
            concept_preds_sigmoid: (N, 312) sigmoid 后的概念预测
            attr_labels: (N, 312) 真实属性标签
            model2: C→Y 预测模型 (MLP)
            concept_groups: dict, 组 ID → 属性索引列表
            ptl_5, ptl_95: 概念预测的百分位值
            max_steps: 最大干预步数
        Returns:
            episode_reward: 本 episode 总奖励
        """
        N = concept_preds_sigmoid.shape[0]
        log_probs_list = []
        rewards_list = []

        for img_id in range(min(N, 32)):  # 限制每 episode 样本数
            attr_preds = concept_preds_sigmoid[img_id].clone()
            intervened_groups = set()

            # 计算初始分类熵
            stage2_input = attr_preds.unsqueeze(0).to(device)
            if use_relu:
                stage2_input = F.relu(stage2_input)
            elif use_sigmoid:
                pass  # 已经是 sigmoid
            initial_probs = F.softmax(model2(stage2_input), dim=1)
            initial_entropy = self.compute_entropy(initial_probs.squeeze().detach().cpu().numpy())

            prev_entropy = initial_entropy
            step_log_probs = []
            step_rewards = []

            for step in range(max_steps):
                # 选择下一个干预组
                group_id, log_prob = self.policy.select_action(
                    attr_preds.detach(), intervened_groups
                )
                intervened_groups.add(group_id)

                # 执行干预：用真实标签替换该组的概念预测
                attr_indices = concept_groups[group_id]
                for attr_idx in attr_indices:
                    attr_preds[attr_idx] = float(attr_labels[img_id, attr_idx])
                    if use_relu or not use_sigmoid:
                        attr_preds[attr_idx] = (1 - attr_labels[img_id, attr_idx]) * ptl_5[attr_idx] \
                            + attr_labels[img_id, attr_idx] * ptl_95[attr_idx]

                # 计算干预后的分类熵
                stage2_input = attr_preds.unsqueeze(0).to(device)
                if use_relu:
                    stage2_input = F.relu(stage2_input)
                new_probs = F.softmax(model2(stage2_input), dim=1)
                new_entropy = self.compute_entropy(new_probs.squeeze().detach().cpu().numpy())

                # 奖励 = 熵降低量
                reward = prev_entropy - new_entropy
                step_log_probs.append(log_prob)
                step_rewards.append(reward)
                prev_entropy = new_entropy

            # 计算折扣奖励
            discounted_rewards = []
            R = 0
            for r in reversed(step_rewards):
                R = r + self.gamma_discount * R
                discounted_rewards.insert(0, R)
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

            log_probs_list.extend(step_log_probs)
            rewards_list.extend(discounted_rewards)

        # REINFORCE 更新
        if len(log_probs_list) == 0:
            return 0.0

        log_probs_tensor = torch.stack(log_probs_list)
        rewards_tensor = torch.stack(rewards_list)

        # 基线：用奖励均值减少方差
        baseline = rewards_tensor.mean()
        advantage = rewards_tensor - baseline

        loss = -(log_probs_tensor * advantage).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return rewards_tensor.mean().item()

    def train(self, concept_preds_sigmoid, attr_labels, model2,
              concept_groups, ptl_5, ptl_95, n_epochs=50,
              max_steps=10, use_relu=False, use_sigmoid=False,
              device='cuda'):
        """多轮训练策略网络。

        Args:
            concept_preds_sigmoid: (N, 312) 验证集概念预测
            attr_labels: (N, 312) 验证集真实属性
            model2: C→Y 模型
            concept_groups: 概念分组
            ptl_5, ptl_95: 百分位值
            n_epochs: 训练轮数
        Returns:
            训练后的策略网络
        """
        concept_preds_sigmoid = torch.tensor(concept_preds_sigmoid, dtype=torch.float32)
        attr_labels = torch.tensor(attr_labels, dtype=torch.float32)

        for epoch in range(n_epochs):
            reward = self.train_episode(
                concept_preds_sigmoid, attr_labels, model2,
                concept_groups, ptl_5, ptl_95,
                max_steps=max_steps, use_relu=use_relu,
                use_sigmoid=use_sigmoid, device=device
            )
            if epoch % 10 == 0:
                print(f"Policy-TTI Epoch {epoch}: avg reward = {reward:.4f}")

        return self.policy


def policy_intervention(policy_net, concept_preds_sigmoid, attr_labels,
                        model2, concept_groups, ptl_5, ptl_95,
                        n_replace, use_relu=False, use_sigmoid=False,
                        device='cuda'):
    """使用学到的策略进行干预评估。

    Args:
        policy_net: 训练好的 PolicyNetwork
        concept_preds_sigmoid: (312,) 单样本概念预测
        attr_labels: (312,) 单样本真实属性
        model2: C→Y 模型
        n_replace: 干预组数
    Returns:
        group_ids: list, 选中的组 ID（按策略顺序）
    """
    attr_preds = concept_preds_sigmoid.clone()
    intervened_groups = []
    all_group_ids = []

    for step in range(n_replace):
        intervened_set = set(intervened_groups)
        logits = policy_net(attr_preds.unsqueeze(0)).squeeze(0)

        # 排除已干预的组
        for g in intervened_set:
            logits[g] = -1e9

        # 贪心选择（推理时不采样，选概率最大的）
        group_id = logits.argmax().item()
        intervened_groups.append(group_id)
        all_group_ids.append(group_id)

        # 执行干预
        attr_indices = concept_groups[group_id]
        for attr_idx in attr_indices:
            attr_preds[attr_idx] = float(attr_labels[attr_idx])
            if use_relu or not use_sigmoid:
                attr_preds[attr_idx] = (1 - attr_labels[attr_idx]) * ptl_5[attr_idx] \
                    + attr_labels[attr_idx] * ptl_95[attr_idx]

    return all_group_ids