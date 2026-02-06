# 训练进度条和时间估计功能

## 🎯 新增功能

为训练过程添加了可视化进度条和时间估计，让训练过程更加直观和可控。

## ✨ 功能特性

### 1. Epoch 内进度条
- 显示当前 batch 的处理进度
- 实时显示损失和准确率
- 显示已用时间和预计剩余时间

### 2. 整体训练进度
- 显示当前 epoch 和总 epoch 数
- 显示每个 epoch 的用时
- 计算并显示完成所有训练的预计剩余时间（ETA）
- 显示总训练时间

### 3. 美化的输出格式
- 清晰的分隔线
- 训练/验证指标对比
- 最佳模型标记（🌟）
- 早停提示（⚠️）

## 📊 输出示例

### Epoch 内进度条
```
Training: 100%|████████████████| 94/94 [00:45<00:00] loss: 0.3245 acc: 89.23%
Validation: 100%|██████████████| 19/19 [00:08<00:00] loss: 0.2891 acc: 91.45%
```

### Epoch 总结
```
────────────────────────────────────────────────────────────────────────────────
Epoch [ 15/100] | Time: 53.2s | ETA: 1:15:20 | Total: 0:13:18
────────────────────────────────────────────────────────────────────────────────
  Train → Loss:  0.3245 | Acc:  89.23%
  Val   → Loss:  0.2891 | Acc:  91.45% 🌟 NEW BEST!
  Best  → Epoch:  15 | Acc:  91.45%
────────────────────────────────────────────────────────────────────────────────
  Current learning rate: 0.010000
```

### 训练完成
```
════════════════════════════════════════════════════════════════════════════════
Training completed!
Total time: 1:28:45
Best validation accuracy: 95.83% at epoch 87
════════════════════════════════════════════════════════════════════════════════
```

## 🔧 技术实现

### 1. 使用 tqdm 库

```python
from tqdm import tqdm

# 创建进度条
pbar = tqdm(enumerate(loader), total=len(loader), 
            desc="Training", leave=False, ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

# 更新进度条信息
pbar.set_postfix({
    'loss': f'{loss_meter.avg:.4f}',
    'acc': f'{acc_meter.avg:.2f}%'
})
```

### 2. 时间估计算法

```python
import time
from datetime import timedelta

# 记录每个 epoch 的时间
epoch_times = []
epoch_start_time = time.time()

# ... 训练代码 ...

epoch_time = time.time() - epoch_start_time
epoch_times.append(epoch_time)

# 计算平均时间和预计剩余时间
avg_epoch_time = sum(epoch_times) / len(epoch_times)
remaining_epochs = total_epochs - (current_epoch + 1)
eta_seconds = avg_epoch_time * remaining_epochs
eta = str(timedelta(seconds=int(eta_seconds)))
```

### 3. 格式化输出

```python
# 使用 Unicode 字符美化输出
print(f"{'─'*80}")  # 分隔线
print(f"Epoch [{epoch+1:3d}/{args.epochs}]")  # 对齐的数字
print(f"  Train → Loss: {train_loss:7.4f}")  # 固定宽度
```

## 📝 修改的文件

### 1. `src/util/train_util.py`

#### 添加导入
```python
from tqdm import tqdm
```

#### 修改 `run_epoch` 函数
- 添加进度条创建
- 在循环中更新进度条
- 显示实时损失和准确率

#### 修改 `run_epoch_simple` 函数
- 同样添加进度条支持

### 2. `src/train.py`

#### 添加导入
```python
import time
from datetime import timedelta
```

#### 修改 `train` 函数
- 添加时间跟踪变量
- 计算每个 epoch 的用时
- 计算预计剩余时间（ETA）
- 美化输出格式
- 添加最佳模型标记
- 添加训练完成总结

## 🎨 输出格式说明

### 进度条格式
```
Training: 100%|████████████████| 94/94 [00:45<00:00]
          ↑     ↑                ↑      ↑      ↑
          描述  进度条            当前/总数  已用时间  剩余时间
```

### Epoch 信息格式
```
Epoch [ 15/100] | Time: 53.2s | ETA: 1:15:20 | Total: 0:13:18
      ↑         ↑              ↑              ↑
      当前/总数  本epoch用时    预计剩余时间    总用时
```

### 指标显示格式
```
  Train → Loss:  0.3245 | Acc:  89.23%
  Val   → Loss:  0.2891 | Acc:  91.45% 🌟 NEW BEST!
  Best  → Epoch:  15 | Acc:  91.45%
```

## 🚀 使用方法

无需修改命令，直接运行即可看到新的进度显示：

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
    -b 64 \
    -weight_decay 0.0004 \
    -lr 0.01 \
    -scheduler_step 20 \
    -dset birds \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011
```

## 💡 优势

### 1. 更好的可视化
- 一目了然地看到训练进度
- 实时监控损失和准确率变化
- 清晰的格式化输出

### 2. 时间管理
- 知道训练还需要多久
- 可以合理安排时间
- 避免长时间等待的焦虑

### 3. 问题诊断
- 快速发现训练异常
- 及时调整超参数
- 监控过拟合情况

### 4. 用户体验
- 专业的输出格式
- 清晰的信息层次
- 重要信息高亮显示

## 🔍 技术细节

### 进度条配置
```python
bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
```
- `{l_bar}`: 左侧标签（描述和百分比）
- `{bar}`: 进度条本身
- `{n_fmt}/{total_fmt}`: 当前/总数
- `{elapsed}`: 已用时间
- `{remaining}`: 预计剩余时间

### 时间格式化
```python
str(timedelta(seconds=int(eta_seconds)))
```
- 自动转换为 `HH:MM:SS` 格式
- 例如: `1:15:20` 表示 1小时15分20秒

### 动态更新
```python
pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.2f}%'})
```
- 实时更新进度条右侧的信息
- 不会产生新行，保持输出整洁

## 📊 性能影响

- **最小开销**: tqdm 的性能开销非常小（< 0.1%）
- **内存友好**: 不会增加显著的内存使用
- **可配置**: 可以通过参数调整更新频率

## 🎯 未来改进

可以考虑添加：
1. 训练曲线实时绘制
2. TensorBoard 集成
3. 邮件/消息通知
4. GPU 使用率监控
5. 自动保存检查点

## 📚 相关资源

- [tqdm 文档](https://tqdm.github.io/)
- [Python datetime 文档](https://docs.python.org/3/library/datetime.html)
- [终端颜色代码](https://en.wikipedia.org/wiki/ANSI_escape_code)

---

现在训练过程更加直观和专业了！🎉
