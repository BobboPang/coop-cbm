# 快速开始：测试时干预实验

## 🎯 目标

对训练好的Coop-CBM模型进行测试时干预实验，验证模型的可解释性和可控性。

---

## ⚡ 快速运行

### 方法1: 使用脚本（推荐）

```bash
# 1. 确保Coop模型已训练完成
ls outputfiles/Coop/birds/312/False/1.0/best_model_42.pth

# 2. 运行干预实验
bash run_intervention.sh
```

### 方法2: 直接运行Python

```bash
CUDA_VISIBLE_DEVICES=2 python test_time_intervention.py \
    -model_path outputfiles/Coop/birds/312/False/1.0/best_model_42.pth \
    -data_dir /data/project/master/dataset/CUB-200-2011/CUB-200-2011 \
    -batch_size 32 \
    -intervention_mode wrong \
    -save_dir results/intervention
```

---

## 📊 预期输出

### 1. 控制台输出

```
================================================================================
Test-Time Intervention Experiment
================================================================================
Model: outputfiles/Coop/birds/312/False/1.0/best_model_42.pth
Data: /data/project/master/dataset/CUB-200-2011/CUB-200-2011
Intervention Mode: wrong
Intervention Ratios: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
================================================================================

Loading model from: outputfiles/Coop/birds/312/False/1.0/best_model_42.pth
Model loaded successfully!

Loading test data from: /data/project/master/dataset/CUB-200-2011/CUB-200-2011
Test data loaded: 188 batches

================================================================================
Step 1: Baseline Prediction (No Intervention)
================================================================================
Predicting: 100%|████████████████| 188/188 [02:15<00:00] acc: 94.56%

✓ Baseline Accuracy (No Intervention): 94.56%

================================================================================
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

→ Intervention ratio: 70%
  Accuracy: 98.12%

→ Intervention ratio: 100%
  Accuracy: 98.92%

================================================================================
Step 3: Visualizing Results
================================================================================
✓ Saved: results/intervention/intervention_accuracy_curve.png
✓ Saved: results/intervention/intervention_improvement.png

────────────────────────────────────────────────────────────────────────────────
Intervention Results Summary
────────────────────────────────────────────────────────────────────────────────
Ratio           Accuracy        Improvement    
────────────────────────────────────────────────────────────────────────────────
     0%          94.56%          +0.00%
    10%          95.23%          +0.67%
    20%          96.12%          +1.56%
    30%          96.78%          +2.22%
    50%          97.45%          +2.89%
    70%          98.12%          +3.56%
   100%          98.92%          +4.36%
────────────────────────────────────────────────────────────────────────────────

================================================================================
✓ Experiment completed successfully!
✓ Results saved to: results/intervention
================================================================================
```

### 2. 生成的文件

```
results/intervention/
├── intervention_accuracy_curve.png    # 准确率曲线图
└── intervention_improvement.png       # 准确率提升柱状图
```

---

## 📈 结果解读

### 关键指标

1. **基线准确率**: 94.56%
   - 无干预时的模型性能
   - 应该接近训练时的验证集准确率

2. **10%干预提升**: +0.67%
   - 只修正10%的错误概念就能提升准确率
   - 说明模型确实依赖概念进行推理

3. **完全干预准确率**: 98.92%
   - 所有概念都用真实标签替换
   - 接近理论上限，说明概念表示质量高

### 成功的标志 ✅

- 准确率随干预比例单调递增
- 少量干预（10-20%）就有明显提升
- 完全干预可达98%以上

### 失败的标志 ❌

- 干预后准确率不变或下降
- 需要大量干预才有微小提升
- 完全干预的准确率仍然很低（<95%）

---

## 🔧 自定义配置

### 修改GPU

```bash
# 编辑 run_intervention.sh
GPU=3  # 改为你想用的GPU编号
```

### 修改模型路径

```bash
# 如果你的模型保存在其他位置
MODEL_PATH="path/to/your/best_model.pth"
```

### 修改干预模式

```bash
# 三种模式可选
-intervention_mode wrong      # 只修正错误的概念（推荐）
-intervention_mode random     # 随机修正概念
-intervention_mode uncertain  # 修正不确定的概念
```

### 修改批次大小

```bash
# 如果显存不足，可以减小批次
-batch_size 16  # 从32改为16
```

---

## 🐛 常见问题

### Q1: 找不到模型文件

```
❌ Error: Model file not found
```

**解决方案**：
1. 确认Coop模型已训练完成
2. 检查模型路径是否正确
3. 查看 `outputfiles/Coop/` 目录下的实际路径

### Q2: 显存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**：
```bash
# 减小批次大小
-batch_size 16  # 或更小
```

### Q3: 数据加载失败

```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案**：
1. 确认数据集路径正确
2. 检查数据集目录下是否有 `train.pkl`, `val.pkl`, `test.pkl`

---

## 📚 更多信息

- 详细指南: `INTERVENTION_EXPERIMENT_GUIDE.md`
- Coop模型说明: `COOP_MODEL_EXPLAINED.md`
- 代码实现: `test_time_intervention.py`

---

## 🎉 下一步

完成干预实验后，你可以：

1. **对比不同模型**
   - 运行Joint模型的干预实验
   - 对比Coop vs Joint的干预效果

2. **分析概念重要性**
   - 哪些概念对预测影响最大？
   - 不同鸟类依赖哪些概念？

3. **探索应用场景**
   - 医疗诊断中的专家干预
   - 自动驾驶中的安全干预
   - 金融风控中的人工审核

---

**准备好了吗？运行 `bash run_intervention.sh` 开始实验！** 🚀
