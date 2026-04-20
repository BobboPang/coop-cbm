# 快速显存优化方案

## 🎯 最简单的解决方案（立即可用）

只需修改一个参数，无需改代码！

### 方法：减小 Batch Size

将 batch size 从 64 改为 **32** 或 **48**：

```bash
# 原命令
python experiments.py CUB Concept_XtoC --seed 42 -b 64 ...

# 优化后（节省约 2-3G 显存）
python experiments.py CUB Concept_XtoC --seed 42 -b 32 ...
```

### 完整命令示例

```bash
python experiments.py CUB Concept_XtoC \
    --seed 42 \
    -log_dir outputs/Concept_XtoC/ \
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
    -b 32 \
    -weight_decay 0.0004 \
    -lr 0.01 \
    -scheduler_step 20 \
    -dset birds \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011
```

## 📊 预期效果

| Batch Size | 显存占用 | 训练时间 | 精度影响 |
|-----------|---------|---------|---------|
| 64（原始） | 14.5G | 1.0x | 基准 |
| 48 | ~11-12G | 1.3x | 无 |
| 32 | ~9-10G | 2.0x | 无 |
| 16 | ~6-7G | 4.0x | 可能轻微下降 |

## 💡 推荐配置

### 如果显存 12G
```bash
-b 48  # 显存占用约 11-12G
```

### 如果显存 10G
```bash
-b 32  # 显存占用约 9-10G
```

### 如果显存 8G
```bash
-b 24  # 显存占用约 7-8G
```

## ⚡ 补偿训练时间增加的方法

虽然 batch size 减小会增加训练时间，但可以通过以下方法补偿：

### 1. 减少 epoch 数（如果只是测试）
```bash
-e 100  # 从 200 改为 100
```

### 2. 增加学习率（小心使用）
```bash
-lr 0.015  # 从 0.01 增加到 0.015（仅当 batch_size >= 32）
```

### 3. 调整学习率衰减
```bash
-scheduler_step 15  # 从 20 改为 15，更快衰减
```

## 🔧 其他简单优化

### 减少 DataLoader Workers
在 `src/data/data_sel.py` 中修改（可选）：

```python
# 找到这行
train_params = {'batch_size': args.batch_size}

# 改为
train_params = {
    'batch_size': args.batch_size,
    'num_workers': 2,  # 添加这行，减少内存占用
    'pin_memory': True
}
```

### 禁用调试信息
在 `src/util/train_util.py` 中注释掉调试打印（已经在代码中完成）：

```python
# 注释掉这行
# print(f"Debug: len(attr_criterion)={len(attr_criterion)}, ...")
```

## ✅ 总结

**最快的解决方案**：
1. 将 `-b 64` 改为 `-b 32`
2. 预期显存：~9-10G（节省 4-5G）
3. 训练时间：增加约 2倍
4. 精度：无影响

**立即可用的完整命令**：
```bash
python experiments.py CUB Concept_XtoC --seed 42 \
    -b 32 \
    -log_dir outputs/Concept_XtoC/ \
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

现在就可以开始训练了！🚀
