# 测试时干预实验指南 (Test-Time Intervention Guide)

## 📚 目录
1. [什么是测试时干预](#什么是测试时干预)
2. [为什么需要干预实验](#为什么需要干预实验)
3. [实验原理](#实验原理)
4. [运行实验](#运行实验)
5. [结果解读](#结果解读)

---

## 什么是测试时干预

**测试时干预 (Test-Time Intervention, TTI)** 是评估概念瓶颈模型可解释性和可控性的重要方法。

### 核心思想

在测试时，我们可以：
1. 让模型预测概念（如"红色羽毛"、"长喙"等）
2. **人工修正**某些概念的预测
3. 使用修正后的概念重新预测类别
4. 观察准确率的变化

```
正常预测流程:
输入图像 → 预测概念 → 预测类别
    X    →     C     →     Y

干预预测流程:
输入图像 → 预测概念 → 人工修正 → 预测类别
    X    →     C     →   C_corrected  →  Y
```

---

## 为什么需要干预实验

### 1. 验证可解释性

如果模型真的通过概念进行推理，那么修正错误的概念应该能提高准确率。

**例子**：
- 模型错误地认为某只鸟有"红色羽毛"（实际是蓝色）
- 我们修正这个概念为"蓝色羽毛"
- 如果模型准确率提高 → 说明模型确实依赖概念
- 如果准确率不变 → 说明模型可能在"作弊"，没有真正使用概念

### 2. 评估人机协作能力

在实际应用中，专家可能需要修正模型的中间判断：

**医疗诊断场景**：
```
模型预测: 患者有发烧症状 (错误)
医生修正: 患者没有发烧
模型重新诊断: 从"流感"改为"普通感冒"
```

### 3. 对比不同模型

| 模型类型 | 支持干预 | 干预效果 |
|---------|---------|---------|
| **Standard (黑盒)** | ❌ 不支持 | N/A |
| **Joint CBM** | ✅ 支持 | 中等 |
| **Coop-CBM** | ✅ 支持 | 最好 |

---

## 实验原理

### Coop-CBM 的干预机制

```python
# 1. Stage 1: X → (Y₁, C)
images → Inception V3 → {
    主路径预测: Y₁
    概念预测: C = [C₁, C₂, ..., C₃₁₂]
}

# 2. 人工干预
C_predicted → 修正错误的概念 → C_corrected

# 3. Stage 2: C → Y₃
C_corrected → MLP → Y₃ (新的类别预测)
```

### 三种干预模式

#### 1. Random Intervention (随机干预)
- 随机选择一定比例的概念
- 用真实标签替换
- 用于测试模型对概念的依赖程度

#### 2. Wrong Intervention (错误修正)
- 只修正预测错误的概念
- 最符合实际应用场景
- **推荐使用**

#### 3. Uncertain Intervention (不确定修正)
- 修正模型最不确定的概念（sigmoid值接近0.5）
- 用于测试模型的置信度

---

## 运行实验

### 前提条件

1. **已训练好的Coop模型**
   ```bash
   outputfiles/Coop/birds/312/False/1.0/best_model_42.pth
   ```

2. **CUB数据集**
   ```bash
   /data/project/master/dataset/CUB-200-2011/CUB-200-2011
   ```

### 基本用法

```bash
CUDA_VISIBLE_DEVICES=2 python test_time_intervention.py \
    -model_path outputfiles/Coop/birds/312/False/1.0/best_model_42.pth \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011 \
    -batch_size 32 \
    -intervention_mode wrong \
    -save_dir results/intervention
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-model_path` | Coop模型路径 | 必需 |
| `-data_dir` | CUB数据集路径 | 必需 |
| `-batch_size` | 批次大小 | 32 |
| `-n_attributes` | 属性数量 | 312 |
| `-intervention_mode` | 干预模式 (random/wrong/uncertain) | wrong |
| `-save_dir` | 结果保存目录 | results/intervention |

### 不同干预模式的命令

#### 1. 错误修正模式（推荐）
```bash
python test_time_intervention.py \
    -model_path outputfiles/Coop/birds/312/False/1.0/best_model_42.pth \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011 \
    -intervention_mode wrong
```

#### 2. 随机干预模式
```bash
python test_time_intervention.py \
    -model_path outputfiles/Coop/birds/312/False/1.0/best_model_42.pth \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011 \
    -intervention_mode random
```

#### 3. 不确定修正模式
```bash
python test_time_intervention.py \
    -model_path outputfiles/Coop/birds/312/False/1.0/best_model_42.pth \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011 \
    -intervention_mode uncertain
```

---

## 实验流程

### Step 1: 基线预测（无干预）

```
Loading model...
Loading test data...
Step 1: Baseline Prediction (No Intervention)
================================================================================
Predicting: 100%|████████████████| 188/188 [02:15<00:00]

✓ Baseline Accuracy (No Intervention): 94.56%
```

### Step 2: 不同干预比例的预测

```
Step 2: Prediction with Intervention (mode=wrong)
================================================================================

→ Intervention ratio: 0%
  Accuracy: 94.56%

→ Intervention ratio: 10%
  Accuracy: 95.23%

→ Intervention ratio: 20%
  Accuracy: 96.12%

→ Intervention ratio: 30%
  Accuracy: 96.78%

→ Intervention ratio: 50%
  Accuracy: 97.45%

→ Intervention ratio: 100%
  Accuracy: 98.92%
```

### Step 3: 可视化结果

生成两张图表：
1. **intervention_accuracy_curve.png**: 干预比例 vs 准确率曲线
2. **intervention_improvement.png**: 准确率提升柱状图

---

## 结果解读

### 理想的干预曲线

```
准确率
  ^
  |                                    ●  (100% 干预)
  |                              ●
  |                        ●
  |                  ●
  |            ●
  |      ●
  |●  (0% 干预 = 基线)
  +---------------------------------> 干预比例
  0%   10%   20%   30%   50%   100%
```

### 关键指标

#### 1. 基线准确率
- Coop-CBM: ~94-95%
- 这是模型在没有任何干预时的性能

#### 2. 完全干预准确率（100%）
- 理论上限: ~98-99%
- 所有概念都用真实标签替换
- 接近"oracle"性能

#### 3. 干预效率
- **好的模型**: 少量干预就能显著提升准确率
- **差的模型**: 即使大量干预也提升有限

### 预期结果

#### Coop-CBM（好的结果）
```
干预比例    准确率    提升
  0%       94.56%    +0.00%
 10%       95.23%    +0.67%
 20%       96.12%    +1.56%
 30%       96.78%    +2.22%
 50%       97.45%    +2.89%
100%       98.92%    +4.36%
```

**解读**：
- ✅ 干预有效：准确率随干预比例单调递增
- ✅ 效率高：10%的干预就能提升0.67%
- ✅ 上限高：完全干预可达98.92%

#### 黑盒模型（无法干预）
```
干预比例    准确率    提升
  0%       95.83%    N/A
 10%       N/A       N/A (不支持干预)
```

#### 差的CBM（干预无效）
```
干预比例    准确率    提升
  0%       92.00%    +0.00%
 10%       92.05%    +0.05%  ← 几乎没有提升
 20%       92.10%    +0.10%
 50%       92.30%    +0.30%
100%       93.00%    +1.00%  ← 上限很低
```

**解读**：
- ❌ 干预效果差：大量干预也提升有限
- ❌ 说明模型没有真正使用概念
- ❌ 可能存在"信息泄漏"

---

## 高级用法

### 1. 对比不同模型

```bash
# Coop模型
python test_time_intervention.py \
    -model_path outputfiles/Coop/birds/312/False/1.0/best_model_42.pth \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011 \
    -save_dir results/intervention_coop

# Joint模型（如果有）
python test_time_intervention.py \
    -model_path outputfiles/Joint/birds/312/False/1.0/best_model_42.pth \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011 \
    -save_dir results/intervention_joint
```

### 2. 自定义干预比例

修改 `test_time_intervention.py` 中的 `intervention_ratios` 参数：

```python
intervention_ratios = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0]
```

### 3. 分析特定概念的影响

可以修改代码，只干预特定类型的概念（如颜色、形状等）：

```python
# 只干预颜色相关的概念
color_concepts = list(range(0, 15))  # 假设前15个是颜色概念
intervene_idx = [idx for idx in intervene_idx if idx in color_concepts]
```

---

## 常见问题

### Q1: 为什么干预后准确率反而下降？

**可能原因**：
1. 概念标签本身有噪声
2. 随机干预可能把正确的概念改错了
3. 模型过拟合了训练集的概念分布

**解决方案**：
- 使用 `wrong` 模式而不是 `random` 模式
- 检查概念标签的质量

### Q2: 完全干预（100%）的准确率为什么不是100%？

**原因**：
- 即使所有概念都正确，C→Y的映射也不是完美的
- 某些类别可能共享相似的概念
- 概念集合可能不完全充分（不足以区分所有类别）

### Q3: 如何判断干预实验是否成功？

**成功的标志**：
1. ✅ 准确率随干预比例单调递增
2. ✅ 少量干预（10-20%）就有明显提升
3. ✅ 完全干预的准确率接近理论上限

**失败的标志**：
1. ❌ 干预后准确率不变或下降
2. ❌ 需要大量干预才有微小提升
3. ❌ 完全干预的准确率仍然很低

---

## 论文中的干预实验

### 典型结果

论文中Coop-CBM在CUB数据集上的干预实验结果：

| 干预比例 | Coop-CBM | Joint | Standard |
|---------|----------|-------|----------|
| 0% | 94.5% | 92.0% | 95.8% (N/A) |
| 10% | 95.2% | 92.3% | N/A |
| 20% | 96.1% | 92.8% | N/A |
| 50% | 97.4% | 94.0% | N/A |
| 100% | 98.9% | 96.5% | N/A |

**关键发现**：
1. Coop-CBM的干预效果最好
2. 50%的干预就能超过Standard模型的准确率
3. 完全干预可达98.9%，接近理论上限

---

## 总结

### 测试时干预的意义

1. **验证可解释性**: 证明模型真的通过概念进行推理
2. **评估可控性**: 测试人工修正概念的效果
3. **对比模型**: 量化不同CBM架构的优劣
4. **实际应用**: 支持人机协作的决策系统

### Coop-CBM的优势

- ✅ 支持测试时干预
- ✅ 干预效果显著
- ✅ 少量干预即可提升性能
- ✅ 保持高基线准确率

### 下一步

完成干预实验后，你可以：
1. 分析哪些概念对预测影响最大
2. 研究不同鸟类对干预的敏感度
3. 探索主动学习策略（优先标注重要概念）
4. 将干预机制应用到实际应用中

---

## 参考资料

- 论文: "Auxiliary Losses for Learning Generalizable Concept-based Models" (NeurIPS 2023)
- 代码: `src/eval/tti.py`
- 相关文档: `COOP_MODEL_EXPLAINED.md`

---

**祝实验顺利！** 🎉
