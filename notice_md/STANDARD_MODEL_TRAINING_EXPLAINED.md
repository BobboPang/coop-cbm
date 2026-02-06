# Standard 模型训练详解

## 概述

Standard 模型是一个**标准的图像分类模型**，直接从图像 (X) 预测类别标签 (Y)，不使用概念属性。它作为整个 Coop-CBM 项目的基线模型。

**模型架构**: X → Y (图像 → 类别)

---

## 1. 训练入口 (experiments.py)

### 命令行调用
```bash
python experiments.py CUB Standard --seed 42 \
    -log_dir outputs/Standard/ \
    -e 100 \
    -optimizer sgd \
    -pretrained \
    -use_aux \
    -b 64 \
    -weight_decay 0.0004 \
    -lr 0.01 \
    -scheduler_step 20 \
    -dset birds \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011
```

### 关键参数说明
- `CUB`: 数据集类型
- `Standard`: 实验类型
- `--seed 42`: 随机种子，保证可复现性
- `-e 100`: 训练 100 个 epoch
- `-optimizer sgd`: 使用 SGD 优化器
- `-pretrained`: 使用 ImageNet 预训练权重
- `-use_aux`: 使用辅助分类器（Inception V3 的特性）
- `-b 64`: batch size = 64
- `-lr 0.01`: 初始学习率
- `-scheduler_step 20`: 每 20 个 epoch 学习率衰减

### 代码流程 (experiments.py)
```python
def run_experiments(dataset, args):
    experiment = args[0].exp
    if experiment == 'Standard':
        train_X_to_y(*args)  # 调用 Standard 模型训练函数
```

---

## 2. 模型定义 (src/train.py)

### train_X_to_y 函数
```python
def train_X_to_y(args):
    # 创建 Standard 模型：X → Y
    model = ModelXtoY(
        pretrained=args.pretrained,  # 使用预训练权重
        freeze=args.freeze,          # 是否冻结底层
        num_classes=N_CLASSES,       # 200 个鸟类类别
        use_aux=args.use_aux         # 使用辅助分类器
    )
    train(model, args)  # 调用通用训练函数
```

### ModelXtoY 定义 (src/model/models.py)
```python
def ModelXtoY(pretrained, freeze, num_classes, use_aux):
    # 返回一个 Inception V3 模型
    return inception_v3(
        pretrained=pretrained,
        freeze=freeze,
        num_classes=num_classes,
        aux_logits=use_aux
    )
```

**关键点**:
- 使用 **Inception V3** 作为骨干网络
- 如果 `pretrained=True`，加载 ImageNet 预训练权重
- 如果 `freeze=True`，冻结除全连接层外的所有层
- `aux_logits=True` 启用辅助分类器（帮助梯度传播）

---

## 3. 数据加载 (src/data/data_sel.py)

### selector 函数
```python
def selector(args):
    # 1. 定义数据增强
    transforms_train, transforms_eval = get_transforms()
    
    # 2. 加载 CUB 数据集
    ds_train = DatasetBirds(args.data_dir, args, transform=transforms_train, train=True)
    ds_val = DatasetBirds(args.data_dir, args, transform=transforms_eval, train=True)
    ds_test = DatasetBirds(args.data_dir, args, transform=transforms_eval, train=False)
    
    # 3. 划分训练集和验证集 (80/20 split)
    splits = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    idx_train, idx_val = next(splits.split(np.zeros(len(ds_train)), ds_train.targets))
    
    # 4. 创建 DataLoader
    trainset = DataLoader(dataset=ds_train, sampler=SubsetRandomSampler(idx_train), batch_size=64)
    validset = DataLoader(dataset=ds_val, sampler=SubsetRandomSampler(idx_val), batch_size=64)
    test_loader = DataLoader(dataset=ds_test, batch_size=64)
    
    return trainset, validset, test_loader
```

### 数据增强 (get_transforms)
```python
def get_transforms():
    # 训练时增强
    transforms_train = Compose([
        pad(fill=(123, 117, 104)),      # 填充到 500x500
        RandomCrop((375, 375)),          # 随机裁剪
        RandomHorizontalFlip(),          # 随机水平翻转
        RandomVerticalFlip(),            # 随机垂直翻转
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 标准化
    ])
    
    # 验证/测试时增强
    transforms_eval = Compose([
        pad(fill=(123, 117, 104)),
        CenterCrop((375, 375)),          # 中心裁剪（不随机）
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transforms_train, transforms_eval
```

**关键点**:
- 训练集使用随机增强（裁剪、翻转）增加数据多样性
- 验证集使用确定性增强（中心裁剪）保证评估一致性
- 使用 ImageNet 的均值和标准差进行归一化

---

## 4. 训练循环 (src/train.py)

### train 函数核心流程

```python
def train(model, args):
    # ========== 1. 准备工作 ==========
    # 加载数据
    trainset, validset, test_loader = selector(args)
    
    # 创建输出目录
    dir = os.path.join('outputfiles', args.exp, args.dset, 
                       str(args.n_attributes), str(args.col), 
                       str(args.attr_loss_weight))
    os.makedirs(dir, exist_ok=True)
    
    # 创建日志记录器
    logger = Logger(os.path.join(dir, 'log.txt'))
    
    # 将模型移到 GPU
    model = model.to(device)
    
    # ========== 2. 定义损失函数 ==========
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
    # Standard 模型不需要属性损失，所以 attr_criterion = None
    
    # ========== 3. 定义优化器 ==========
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,              # 学习率 0.01
        momentum=0.9,            # 动量
        weight_decay=args.weight_decay  # 权重衰减 0.0004
    )
    
    # ========== 4. 定义学习率调度器 ==========
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,  # 每 20 个 epoch
        gamma=0.1                        # 学习率乘以 0.1
    )
    # 学习率变化: 0.01 → 0.001 (epoch 20) → 0.0001 (epoch 40) → ...
    
    # ========== 5. 训练循环 ==========
    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0
    
    for epoch in range(0, args.epochs):  # 100 个 epoch
        # --- 训练阶段 ---
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        
        train_loss_meter, train_acc_meter = run_epoch(
            model, optimizer, trainset,
            train_loss_meter, train_acc_meter,
            criterion, attr_criterion=None,  # Standard 不用属性损失
            args, is_training=True
        )
        
        # --- 验证阶段 ---
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        
        with torch.no_grad():  # 验证时不计算梯度
            val_loss_meter, val_acc_meter = run_epoch(
                model, optimizer, validset,
                val_loss_meter, val_acc_meter,
                criterion, attr_criterion=None,
                args, is_training=False
            )
        
        # --- 保存最佳模型 ---
        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(dir, 'best_model_%d.pth' % args.seed))
        
        # --- 记录日志 ---
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                    'Val loss: %.4f\tVal acc: %.4f\tBest val epoch: %d\n'
                    % (epoch, train_loss_meter.avg, train_acc_meter.avg,
                       val_loss_meter.avg, val_acc_meter.avg, best_val_epoch))
        logger.flush()
        
        # --- 更新学习率 ---
        if epoch <= stop_epoch:
            scheduler.step()
        
        # --- 早停机制 ---
        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 300:
            print("Early stopping because acc hasn't improved for a long time")
            break
```

---

## 5. 单个 Epoch 训练 (src/util/train_util.py)

### run_epoch 函数详解

```python
def run_epoch(model, optimizer, loader, loss_meter, acc_meter, 
              criterion, attr_criterion, args, is_training):
    """
    运行一个 epoch 的训练或验证
    """
    # ========== 1. 设置模型模式 ==========
    if is_training:
        model.train()  # 训练模式：启用 Dropout、BatchNorm 更新
    else:
        model.eval()   # 评估模式：禁用 Dropout、BatchNorm 不更新
    
    # ========== 2. 遍历数据批次 ==========
    for _, data in enumerate(loader):
        # 解包数据
        inputs, labels, attr_labels = data  # Standard 模型忽略 attr_labels
        
        # 转换为 Variable 并移到 GPU
        inputs_var = torch.autograd.Variable(inputs).to(device)
        labels_var = torch.autograd.Variable(labels).to(device)
        
        # ========== 3. 前向传播 ==========
        if is_training and args.use_aux:
            # 训练时使用辅助分类器
            # Inception V3 返回: (主输出, 辅助输出)
            outputs, aux_outputs = model(inputs_var)
            
            # 计算损失：主损失 + 0.4 * 辅助损失
            loss_main = 1.0 * criterion(outputs[0], labels_var)
            loss_aux = 0.4 * criterion(aux_outputs[0], labels_var)
            total_loss = loss_main + loss_aux
        else:
            # 验证时只有主输出
            outputs = model(inputs_var)
            total_loss = criterion(outputs[0], labels_var)
        
        # ========== 4. 计算准确率 ==========
        acc = accuracy(outputs[0], labels, topk=(1,))
        acc_meter.update(acc[0], inputs.size(0))
        loss_meter.update(total_loss.item(), inputs.size(0))
        
        # ========== 5. 反向传播（仅训练时）==========
        if is_training:
            optimizer.zero_grad()    # 清空梯度
            total_loss.backward()    # 反向传播计算梯度
            optimizer.step()         # 更新参数
    
    return loss_meter, acc_meter
```

### 关键概念解释

#### 辅助分类器 (Auxiliary Classifier)
```python
# Inception V3 的特殊设计
# 在网络中间层添加额外的分类器，帮助梯度传播

if args.use_aux:
    # 主分类器损失权重 = 1.0
    # 辅助分类器损失权重 = 0.4
    total_loss = 1.0 * main_loss + 0.4 * aux_loss
```

**作用**:
- 缓解梯度消失问题
- 提供额外的正则化
- 加速训练收敛

#### 损失函数
```python
criterion = torch.nn.CrossEntropyLoss()

# 交叉熵损失 = -log(P(正确类别))
# 对于 200 个类别的分类任务
# 输出: [batch_size, 200] 的 logits
# 标签: [batch_size] 的类别索引 (0-199)
```

---

## 6. 模型架构详解 (Inception V3)

### 网络结构
```
输入图像 (3 x 375 x 375)
    ↓
[卷积层组 1] Conv2d_1a_3x3, Conv2d_2a_3x3, Conv2d_2b_3x3
    ↓
[卷积层组 2] Conv2d_3b_1x1, Conv2d_4a_3x3
    ↓
[Inception 模块] Mixed_5b, Mixed_5c, Mixed_5d
    ↓
[Inception 模块] Mixed_6a, Mixed_6b, Mixed_6c, Mixed_6d, Mixed_6e
    ↓
[辅助分类器] ← 如果 use_aux=True，在这里输出辅助预测
    ↓
[Inception 模块] Mixed_7a, Mixed_7b, Mixed_7c
    ↓
[全局平均池化] AdaptiveAvgPool2d
    ↓
[Dropout] p=0.5
    ↓
[全连接层] fc: 2048 → 200 (鸟类类别数)
    ↓
输出 logits (200 维)
```

### 预训练权重
```python
if pretrained:
    # 加载 ImageNet 预训练权重
    model.load_partial_state_dict(torch.load('inception_v3_google.pth'))
    
    if freeze:
        # 冻结除 fc 层外的所有参数
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
```

**迁移学习策略**:
1. 使用 ImageNet 预训练权重初始化
2. 底层特征（边缘、纹理）已经学好
3. 只需微调顶层特征和分类器
4. 大大减少训练时间和所需数据量

---

## 7. 优化策略

### 学习率调度
```python
# 初始学习率: 0.01
# 每 20 个 epoch 衰减为原来的 0.1

Epoch 0-19:   lr = 0.01
Epoch 20-39:  lr = 0.001
Epoch 40-59:  lr = 0.0001
Epoch 60-79:  lr = 0.00001
...
```

**原理**:
- 初期大学习率快速收敛
- 后期小学习率精细调整
- 避免在最优解附近震荡

### 权重衰减 (Weight Decay)
```python
weight_decay = 0.0004

# L2 正则化: Loss = CrossEntropy + 0.0004 * ||weights||^2
```

**作用**:
- 防止过拟合
- 鼓励权重值较小
- 提高模型泛化能力

### 动量 (Momentum)
```python
momentum = 0.9

# 更新规则:
# v_t = 0.9 * v_{t-1} + gradient
# weight = weight - lr * v_t
```

**作用**:
- 加速收敛
- 减少震荡
- 帮助跳出局部最优

---

## 8. 训练监控

### 指标记录
```python
# 每个 epoch 记录:
- Train loss: 训练集损失
- Train accuracy: 训练集准确率
- Val loss: 验证集损失
- Val accuracy: 验证集准确率
- Best val epoch: 最佳验证准确率对应的 epoch
```

### 模型保存
```python
# 只保存验证集上表现最好的模型
if best_val_acc < val_acc_meter.avg:
    best_val_epoch = epoch
    best_val_acc = val_acc_meter.avg
    torch.save(model, 'best_model_42.pth')
```

### 早停机制
```python
# 条件 1: 准确率太低
if epoch >= 100 and val_acc_meter.avg < 3:
    break

# 条件 2: 长时间没有提升
if epoch - best_val_epoch >= 300:
    break
```

---

## 9. 训练结果

### 你的模型性能
```
验证集 Top-1 准确率: 95.83%
验证集 Top-5 准确率: 99.08%
```

### 性能分析
- **95.83%** 的 Top-1 准确率表示模型在 CUB-200 数据集上表现优秀
- **99.08%** 的 Top-5 准确率表示正确答案几乎总在前 5 个预测中
- 这为后续的概念瓶颈模型提供了强有力的基线

---

## 10. 与其他模型的对比

### Standard vs Concept-based Models

| 模型 | 架构 | 可解释性 | 准确率 | 干预能力 |
|------|------|----------|--------|----------|
| **Standard** | X → Y | ❌ 黑盒 | 高 | ❌ 无 |
| **Concept_XtoC** | X → C | ✅ 概念 | - | ❌ 无 |
| **Joint** | X → C → Y | ✅ 概念 | 中 | ✅ 有 |
| **Coop-CBM** | X → (C, Y) | ✅ 概念 | 高 | ✅ 有 |

**Standard 模型的特点**:
- ✅ 准确率最高（没有概念瓶颈的信息损失）
- ❌ 完全黑盒，无法解释预测原因
- ❌ 无法进行测试时干预
- ✅ 作为性能上限的基线

---

## 11. 关键代码总结

### 完整训练流程
```python
# 1. 创建模型
model = ModelXtoY(pretrained=True, freeze=False, 
                  num_classes=200, use_aux=True)

# 2. 加载数据
trainset, validset, test_loader = selector(args)

# 3. 定义损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, 
                      momentum=0.9, weight_decay=0.0004)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

# 4. 训练循环
for epoch in range(100):
    # 训练
    for images, labels, _ in trainset:
        outputs = model(images)
        loss = criterion(outputs[0], labels)
        loss.backward()
        optimizer.step()
    
    # 验证
    with torch.no_grad():
        for images, labels, _ in validset:
            outputs = model(images)
            acc = accuracy(outputs[0], labels)
    
    # 保存最佳模型
    if acc > best_acc:
        torch.save(model, 'best_model.pth')
    
    # 更新学习率
    scheduler.step()
```

---

## 12. 下一步

完成 Standard 模型训练后，接下来应该训练：

1. **Concept_XtoC**: 学习图像到概念的映射
2. **Joint**: 端到端的概念瓶颈模型
3. **Coop-CBM**: 协作式概念瓶颈模型（论文核心贡献）

这样可以对比不同方法在准确率和可解释性之间的权衡。
