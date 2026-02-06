# 显存优化指南：从 14.5G 降到 12G 以下

## 🎯 目标
将训练时的显存占用从 14.5G 降低到 12G 以下（节省约 2.5G）

## 📊 显存占用分析

典型的显存占用分布：
- **模型参数**: ~2-3G (Inception V3)
- **优化器状态**: ~2-3G (SGD with momentum)
- **激活值**: ~4-6G (前向传播中间结果)
- **梯度**: ~2-3G (反向传播)
- **数据批次**: ~1-2G (batch_size=64)

## ✅ 优化策略

### 策略 1: 减小 Batch Size（最有效）⭐⭐⭐⭐⭐

**效果**: 可节省 1-2G 显存

```bash
# 原来: batch_size=64
-b 64

# 优化: batch_size=32
-b 32
```

**优点**:
- 立即生效，显存占用减半
- 实现简单，无需修改代码

**缺点**:
- 训练时间增加约 2倍
- 可能需要调整学习率

**补偿措施**:
```bash
# 使用梯度累积来保持等效 batch size
# 在代码中实现梯度累积（见下文）
```

---

### 策略 2: 启用混合精度训练（推荐）⭐⭐⭐⭐⭐

**效果**: 可节省 30-50% 显存（约 4-7G）

#### 实现方法

修改 `src/train.py`:

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train(model, args):
    # ... 现有代码 ...
    
    # 添加混合精度训练
    use_amp = True  # 启用自动混合精度
    scaler = GradScaler() if use_amp else None
    
    # 在训练循环中使用
    for epoch in range(args.epochs):
        if use_amp:
            train_loss_meter, train_acc_meter = run_epoch_amp(
                model, optimizer, train_loader, train_loss_meter, 
                train_acc_meter, criterion, attr_criterion, args, 
                scaler, is_training=True
            )
        else:
            # 原来的代码
            train_loss_meter, train_acc_meter = run_epoch(...)
```

修改 `src/util/train_util.py`:

```python
from torch.cuda.amp import autocast

def run_epoch_amp(model, optimizer, loader, loss_meter, acc_meter, 
                  criterion, attr_criterion, args, scaler, is_training):
    """支持混合精度的训练函数"""
    if is_training:
        model.train()
    else:
        model.eval()
    
    desc = "Training" if is_training else "Validation"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, 
                leave=False, ncols=100)
    
    for batch_idx, data in pbar:
        # ... 数据加载代码 ...
        
        # 使用自动混合精度
        with autocast():
            outputs = model(inputs_var)
            # ... 计算损失 ...
            total_loss = sum(losses)
        
        # 计算准确率（不需要混合精度）
        acc = accuracy(outputs[0], labels, topk=(1,))
        acc_meter.update(acc[0], inputs.size(0))
        loss_meter.update(total_loss.item(), inputs.size(0))
        
        if is_training:
            optimizer.zero_grad()
            # 使用 scaler 进行反向传播
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 
                         'acc': f'{acc_meter.avg:.2f}%'})
    
    return loss_meter, acc_meter
```

**优点**:
- 显存节省显著
- 训练速度可能更快
- 精度损失很小（< 0.1%）

**缺点**:
- 需要修改代码
- 需要 PyTorch >= 1.6

---

### 策略 3: 梯度累积（保持性能）⭐⭐⭐⭐

**效果**: 配合小 batch size 使用，节省 1-2G 显存

修改 `src/util/train_util.py`:

```python
def run_epoch(model, optimizer, loader, loss_meter, acc_meter, 
              criterion, attr_criterion, args, is_training):
    if is_training:
        model.train()
    else:
        model.eval()
    
    # 梯度累积参数
    accumulation_steps = 2  # 累积2个batch再更新
    
    desc = "Training" if is_training else "Validation"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, 
                leave=False, ncols=100)
    
    for batch_idx, data in pbar:
        # ... 数据加载和前向传播 ...
        
        # 计算损失
        total_loss = sum(losses) / accumulation_steps  # 除以累积步数
        
        loss_meter.update(total_loss.item() * accumulation_steps, inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))
        
        if is_training:
            # 反向传播（累积梯度）
            total_loss.backward()
            
            # 每 accumulation_steps 步更新一次参数
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 
                         'acc': f'{acc_meter.avg:.2f}%'})
    
    # 处理最后不足 accumulation_steps 的batch
    if is_training and (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return loss_meter, acc_meter
```

**使用方法**:
```bash
# batch_size=32, accumulation_steps=2
# 等效于 batch_size=64
python experiments.py CUB Concept_XtoC -b 32 ...
```

---

### 策略 4: 梯度检查点（Gradient Checkpointing）⭐⭐⭐

**效果**: 可节省 2-3G 显存

修改 `src/model/template_model.py`:

```python
import torch.utils.checkpoint as checkpoint

class Inception3(nn.Module):
    def __init__(self, ..., use_checkpoint=False):
        super(Inception3, self).__init__()
        self.use_checkpoint = use_checkpoint
        # ... 其他初始化 ...
    
    def forward(self, x):
        # ... 前面的层 ...
        
        # 对计算密集的层使用 checkpoint
        if self.use_checkpoint and self.training:
            x = checkpoint.checkpoint(self.Mixed_5b, x)
            x = checkpoint.checkpoint(self.Mixed_5c, x)
            x = checkpoint.checkpoint(self.Mixed_5d, x)
            # ... 其他层 ...
        else:
            x = self.Mixed_5b(x)
            x = self.Mixed_5c(x)
            x = self.Mixed_5d(x)
            # ... 其他层 ...
        
        return x
```

**优点**:
- 显存节省明显
- 不影响精度

**缺点**:
- 训练时间增加 10-20%
- 需要修改模型代码

---

### 策略 5: 减少 DataLoader Workers⭐⭐

**效果**: 可节省 0.5-1G 显存

修改 `src/data/data_sel.py`:

```python
def selector(args):
    # ... 现有代码 ...
    
    # 减少 num_workers
    train_params = {
        'batch_size': args.batch_size,
        'num_workers': 2,  # 从 4 改为 2
        'pin_memory': True
    }
    
    trainset = td.DataLoader(dataset=ds_train, 
                            sampler=td.SubsetRandomSampler(idx_train), 
                            **train_params)
    # ...
```

---

### 策略 6: 清理不必要的变量⭐⭐

修改 `src/util/train_util.py`:

```python
def run_epoch(model, optimizer, loader, loss_meter, acc_meter, 
              criterion, attr_criterion, args, is_training):
    # ... 训练代码 ...
    
    if is_training:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 清理中间变量
        del total_loss, outputs, losses
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

### 策略 7: 禁用辅助分类器（训练后期）⭐⭐⭐

**效果**: 可节省 1-2G 显存

修改 `src/train.py`:

```python
def train(model, args):
    # ... 现有代码 ...
    
    for epoch in range(args.epochs):
        # 前 50% epoch 使用辅助分类器
        # 后 50% epoch 禁用以节省显存
        use_aux_this_epoch = epoch < (args.epochs // 2)
        
        if not use_aux_this_epoch:
            model.aux_logits = False
        
        # ... 训练代码 ...
```

---

## 🚀 推荐的组合方案

### 方案 A: 快速实现（无需修改代码）

```bash
python experiments.py CUB Concept_XtoC \
    --seed 42 \
    -b 32 \  # 从 64 改为 32（节省 1-2G）
    -e 200 \
    -optimizer sgd \
    -pretrained \
    -use_aux \
    -use_attr \
    -bottleneck \
    -weighted_loss multiple \
    -n_attributes 312 \
    -attr_loss_weight 1.0 \
    -normalize_loss \
    -weight_decay 0.0004 \
    -lr 0.01 \
    -scheduler_step 20 \
    -dset birds \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011
```

**预期显存**: ~12-13G
**训练时间**: 增加约 2倍

---

### 方案 B: 最佳性能（推荐）⭐⭐⭐⭐⭐

1. **减小 batch size**: 64 → 32
2. **启用梯度累积**: accumulation_steps=2
3. **启用混合精度训练**: AMP

**预期显存**: ~8-10G
**训练时间**: 与原来相当或更快
**精度**: 几乎无损失

---

### 方案 C: 极限优化

1. **batch size**: 64 → 16
2. **梯度累积**: accumulation_steps=4
3. **混合精度训练**: AMP
4. **梯度检查点**: 启用
5. **减少 workers**: 4 → 2

**预期显存**: ~6-8G
**训练时间**: 增加约 20-30%
**精度**: 损失 < 0.5%

---

## 💻 快速实现代码

### 1. 创建优化版的 train_util.py

```python
# src/util/train_util_optimized.py
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def run_epoch_optimized(model, optimizer, loader, loss_meter, acc_meter, 
                       criterion, attr_criterion, args, scaler, is_training):
    """
    优化版训练函数：支持混合精度 + 梯度累积
    """
    if is_training:
        model.train()
    else:
        model.eval()
    
    accumulation_steps = getattr(args, 'accumulation_steps', 1)
    use_amp = scaler is not None
    
    desc = "Training" if is_training else "Validation"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=desc, 
                leave=False, ncols=100)
    
    for batch_idx, data in pbar:
        # 数据加载
        if attr_criterion is None:
            inputs, labels, attr_labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            attr_labels = torch.stack([i.long() for i in attr_labels]).float()
            attr_labels_var = attr_labels.to(device)
        
        inputs_var = inputs.to(device)
        labels_var = labels.to(device)
        
        # 混合精度前向传播
        with autocast(enabled=use_amp):
            if is_training and args.use_aux:
                p_out, outputs, aux_outputs = model(inputs_var)
                # ... 计算损失 ...
            else:
                outputs = model(inputs_var)
                # ... 计算损失 ...
            
            total_loss = sum(losses) / accumulation_steps
        
        # 计算准确率
        if args.bottleneck:
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
        else:
            acc = accuracy(outputs[0], labels, topk=(1,))
        
        acc_meter.update(acc[0] if isinstance(acc, list) else acc, inputs.size(0))
        loss_meter.update(total_loss.item() * accumulation_steps, inputs.size(0))
        
        if is_training:
            # 混合精度反向传播
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.2f}%'
        })
    
    # 处理最后的batch
    if is_training and (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    return loss_meter, acc_meter
```

### 2. 修改 train.py 使用优化版本

```python
# src/train.py
from torch.cuda.amp import GradScaler

def train(model, args):
    # ... 现有代码 ...
    
    # 启用混合精度训练
    use_amp = getattr(args, 'use_amp', True)  # 默认启用
    scaler = GradScaler() if use_amp else None
    
    # 设置梯度累积步数
    args.accumulation_steps = getattr(args, 'accumulation_steps', 2)
    
    print(f"Memory optimization enabled:")
    print(f"  - Mixed Precision: {use_amp}")
    print(f"  - Gradient Accumulation: {args.accumulation_steps} steps")
    print(f"  - Effective Batch Size: {args.batch_size * args.accumulation_steps}")
    
    # ... 训练循环 ...
```

### 3. 添加命令行参数

修改 `src/util/utils.py`:

```python
parser.add_argument('-use_amp', action='store_true', default=True,
                   help='Use automatic mixed precision training')
parser.add_argument('-accumulation_steps', type=int, default=1,
                   help='Gradient accumulation steps')
```

---

## 📊 预期效果对比

| 方案 | Batch Size | 其他优化 | 显存占用 | 训练时间 | 精度损失 |
|------|-----------|---------|---------|---------|---------|
| **原始** | 64 | 无 | 14.5G | 1.0x | 0% |
| **方案A** | 32 | 无 | 12-13G | 2.0x | 0% |
| **方案B** | 32 | AMP + 累积 | 8-10G | 1.0-1.2x | <0.1% |
| **方案C** | 16 | AMP + 累积 + 检查点 | 6-8G | 1.2-1.3x | <0.5% |

---

## 🎯 立即可用的命令

### 最简单（只改 batch size）
```bash
python experiments.py CUB Concept_XtoC --seed 42 \
    -b 32 \
    -log_dir outputs/Concept_XtoC/ -e 200 -optimizer sgd \
    -pretrained -use_aux -use_attr -bottleneck \
    -weighted_loss multiple -n_attributes 312 \
    -attr_loss_weight 1.0 -normalize_loss \
    -weight_decay 0.0004 -lr 0.01 -scheduler_step 20 \
    -dset birds \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011
```

### 推荐（需要实现混合精度）
```bash
python experiments.py CUB Concept_XtoC --seed 42 \
    -b 32 \
    -use_amp \
    -accumulation_steps 2 \
    -log_dir outputs/Concept_XtoC/ -e 200 -optimizer sgd \
    -pretrained -use_aux -use_attr -bottleneck \
    -weighted_loss multiple -n_attributes 312 \
    -attr_loss_weight 1.0 -normalize_loss \
    -weight_decay 0.0004 -lr 0.01 -scheduler_step 20 \
    -dset birds \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011
```

---

## ✅ 总结

**最快的方法**（无需改代码）:
- 将 `-b 64` 改为 `-b 32`
- 预期显存: ~12G

**最推荐的方法**（需要少量代码修改）:
- batch_size=32 + 混合精度训练 + 梯度累积
- 预期显存: ~8-10G
- 训练时间几乎不变
- 精度几乎无损失

现在就可以开始优化了！🚀
