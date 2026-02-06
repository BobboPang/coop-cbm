# 损失函数配置问题修复

## 🐛 问题描述

模型输出维度是 `[64, 1]`，但使用了 `CrossEntropyLoss`，导致维度不匹配。

## 🔍 根本原因

### 模型输出配置
```python
# src/model/template_model.py 第 180 行
for i in range(self.n_attributes):
    self.all_fc.append(FC(2048, 1, expand_dim))  # ← 输出维度 = 1
```

每个属性输出一个 logit 值，形状为 `[batch_size, 1]`

### 损失函数选择
```python
# src/train.py 第 81-86 行
if args.weighted_loss:
    # ✅ 正确：BCEWithLogitsLoss 用于单输出二分类
    attr_criterion.append(torch.nn.BCEWithLogitsLoss())
else:
    # ❌ 错误：CrossEntropyLoss 用于多类分类
    attr_criterion.append(torch.nn.CrossEntropyLoss())
```

### 两种损失函数的区别

| 损失函数 | 输入形状 | 标签形状 | 标签类型 | 用途 |
|---------|---------|---------|---------|------|
| **BCEWithLogitsLoss** | `[N, 1]` 或 `[N]` | `[N]` | Float (0.0/1.0) | 二分类（单输出） |
| **CrossEntropyLoss** | `[N, C]` | `[N]` | Long (0 到 C-1) | 多分类 |

## ✅ 解决方案 1：使用 weighted_loss 参数（推荐）

在运行命令时添加 `-weighted_loss multiple` 参数：

```bash
python experiments.py CUB Concept_XtoC --seed 42 \
    -log_dir outputs/Concept_XtoC/ \
    -e 200 \
    -optimizer sgd \
    -pretrained \
    -use_aux \
    -use_attr \
    -n_attributes 312 \
    -attr_loss_weight 1.0 \
    -normalize_loss \
    -weighted_loss multiple \  # ← 添加这个参数
    -b 64 \
    -weight_decay 0.0004 \
    -lr 0.01 \
    -scheduler_step 20 \
    -dset birds \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011
```

## ✅ 解决方案 2：修改默认损失函数（备选）

如果不想每次都添加参数，可以修改代码默认使用 BCEWithLogitsLoss：

```python
# src/train.py 第 81-86 行
if args.use_attr and not args.no_img:
    attr_criterion = []
    # 修改：默认使用 BCEWithLogitsLoss
    for i in range(args.n_attributes):
        attr_criterion.append(torch.nn.BCEWithLogitsLoss())
```

## 📊 BCEWithLogitsLoss 详解

### 工作原理
```python
# BCEWithLogitsLoss = Sigmoid + BCELoss
# 输入: logit (未经过 sigmoid 的原始输出)
# 输出: 二元交叉熵损失

# 数学公式:
# loss = -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]

# 示例:
logit = model(image)  # [64, 1], 例如: [[2.3], [-1.5], ...]
label = torch.tensor([1.0, 0.0, ...])  # [64]

criterion = nn.BCEWithLogitsLoss()
loss = criterion(logit.squeeze(), label)  # 需要 squeeze 去掉维度 1
```

### 为什么使用 BCEWithLogitsLoss 而不是 BCELoss？

```python
# ❌ 不稳定：分两步
sigmoid_output = torch.sigmoid(logit)
loss = nn.BCELoss()(sigmoid_output, label)

# ✅ 稳定：数值稳定的实现
loss = nn.BCEWithLogitsLoss()(logit, label)
```

BCEWithLogitsLoss 在数值上更稳定，避免了 log(0) 的问题。

## 🔧 代码修复

### 当前的 train_util.py 需要调整

```python
# src/util/train_util.py 第 105 行
# 当前代码
losses.append(args.attr_loss_weight * (
    1.0 * attr_criterion[i](
        outputs[i+out_start].squeeze().type(torch.FloatTensor).to(device), 
        attr_labels_var[:, i]
    ) + 0.4 * attr_criterion[i](
        aux_outputs[i+out_start].squeeze().type(torch.FloatTensor).to(device), 
        attr_labels_var[:, i]
    )
))
```

### 问题分析

1. **squeeze()** 是必要的：
   - 输出: `[64, 1]` → squeeze → `[64]`
   - BCEWithLogitsLoss 期望输入 `[64]`

2. **type(torch.FloatTensor)** 是必要的：
   - BCEWithLogitsLoss 期望 Float 类型
   - 标签也应该是 Float 类型

3. **标签类型**：
   ```python
   # 确保标签是 Float 类型
   attr_labels_var = attr_labels_var.float()
   ```

### 完整的修复代码

```python
# src/util/train_util.py
if attr_criterion is not None and args.attr_loss_weight > 0:
    for i in range(args.n_attributes):
        # 确保标签是 Float 类型（BCEWithLogitsLoss 要求）
        attr_label = attr_labels_var[:, i].float()
        
        # 计算损失
        losses.append(args.attr_loss_weight * (
            1.0 * attr_criterion[i](
                outputs[i+out_start].squeeze(),  # [64, 1] → [64]
                attr_label
            ) + 0.4 * attr_criterion[i](
                aux_outputs[i+out_start].squeeze(),
                attr_label
            )
        ))
```

## 🧪 验证

### 检查损失函数类型
```python
# 添加调试信息
print(f"Attribute criterion type: {type(attr_criterion[0])}")
print(f"Output shape: {outputs[out_start].shape}")
print(f"Label shape: {attr_labels_var[:, 0].shape}")
print(f"Label dtype: {attr_labels_var.dtype}")
```

### 预期输出
```
Attribute criterion type: <class 'torch.nn.modules.loss.BCEWithLogitsLoss'>
Output shape: torch.Size([64, 1])
Label shape: torch.Size([64])
Label dtype: torch.float32
```

## 📝 总结

### 问题根源
- 模型输出 `[64, 1]` 用于二分类
- 错误使用了 `CrossEntropyLoss`（多分类损失）
- 应该使用 `BCEWithLogitsLoss`（二分类损失）

### 解决方法
1. **推荐**: 添加 `-weighted_loss multiple` 参数
2. **备选**: 修改代码默认使用 BCEWithLogitsLoss

### 关键点
- BCEWithLogitsLoss 期望输入 `[N]` 或 `[N, 1]`，标签 `[N]` (Float)
- CrossEntropyLoss 期望输入 `[N, C]`，标签 `[N]` (Long)
- 使用 `.squeeze()` 将 `[64, 1]` 转换为 `[64]`
- 确保标签是 Float 类型

## 🚀 正确的运行命令

```bash
# Concept_XtoC
python experiments.py CUB Concept_XtoC --seed 42 \
    -log_dir outputs/Concept_XtoC/ \
    -e 200 \
    -optimizer sgd \
    -pretrained \
    -use_aux \
    -use_attr \
    -n_attributes 312 \
    -attr_loss_weight 1.0 \
    -normalize_loss \
    -weighted_loss multiple \
    -b 64 \
    -weight_decay 0.0004 \
    -lr 0.01 \
    -scheduler_step 20 \
    -dset birds \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011

# Joint
python experiments.py CUB Joint --seed 42 \
    -log_dir outputs/Joint/ \
    -e 300 \
    -optimizer sgd \
    -pretrained \
    -use_aux \
    -use_attr \
    -n_attributes 312 \
    -attr_loss_weight 1.0 \
    -normalize_loss \
    -weighted_loss multiple \
    -b 64 \
    -weight_decay 0.0004 \
    -lr 0.01 \
    -scheduler_step 20 \
    -dset birds \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011

# Coop
python experiments.py CUB Coop --seed 42 \
    -log_dir outputs/Coop/ \
    -e 300 \
    -optimizer sgd \
    -pretrained \
    -use_aux \
    -use_attr \
    -n_attributes 312 \
    -attr_loss_weight 1.0 \
    -normalize_loss \
    -weighted_loss multiple \
    -b 64 \
    -weight_decay 0.0004 \
    -lr 0.01 \
    -scheduler_step 20 \
    -dset birds \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011
```

现在应该可以正常训练了！
