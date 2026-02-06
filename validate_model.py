"""
验证训练好的模型并可视化预测结果
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 导入项目模块
from src.data.data_sel import selector
from src.util.config import BASE_DIR, N_CLASSES
from analysis import AverageMeter, accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path):
    """加载训练好的模型"""
    print(f"Loading model from: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model

def get_class_names(data_dir):
    """获取CUB数据集的类别名称"""
    class_file = os.path.join(data_dir, 'classes.txt')
    class_names = {}
    try:
        with open(class_file, 'r') as f:
            for line in f:
                idx, name = line.strip().split(' ', 1)
                class_names[int(idx) - 1] = name.split('.')[1]  # 去掉编号前缀
    except FileNotFoundError:
        print(f"Warning: {class_file} not found. Using generic class names.")
        # 使用通用类别名称
        for i in range(200):
            class_names[i] = f"Class_{i}"
    return class_names

def evaluate_model(model, data_loader, class_names):
    """在验证集上评估模型"""
    model.eval()
    
    # 准备统计指标
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()
    
    all_preds = []
    all_labels = []
    all_images = []
    all_probs = []
    
    print("Evaluating model on validation set...")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # 数据加载器返回 (images, labels, attributes)
            images, labels, _ = data
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[0]  # 取分类输出
            
            # 计算准确率
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1.item(), images.size(0))
            top5_acc.update(acc5.item(), images.size(0))
            
            # 保存预测结果（只保存前几个batch用于可视化）
            if batch_idx < 5:
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.topk(1, 1, True, True)
                
                all_images.extend(images.cpu())
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(data_loader)}] "
                      f"Top-1 Acc: {top1_acc.avg:.2f}% "
                      f"Top-5 Acc: {top5_acc.avg:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Final Results on Validation Set:")
    print(f"Top-1 Accuracy: {top1_acc.avg:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc.avg:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        'top1_acc': top1_acc.avg,
        'top5_acc': top5_acc.avg,
        'images': all_images[:16],  # 只保留16张图片用于可视化
        'labels': all_labels[:16],
        'preds': all_preds[:16],
        'probs': all_probs[:16]
    }

def visualize_predictions(results, class_names, save_path='results/standard_predictions.png'):
    """可视化模型预测结果"""
    images = results['images']
    labels = results['labels']
    preds = results['preds']
    probs = results['probs']
    
    # 创建图像网格
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Standard Model Predictions on Validation Set', fontsize=16, y=0.995)
    
    # 反归一化图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            break
        
        # 反归一化
        img = images[idx].numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # 显示图像
        ax.imshow(img)
        
        # 获取预测和真实标签
        true_label = labels[idx]
        pred_label = preds[idx]
        confidence = probs[idx][pred_label] * 100
        
        # 判断预测是否正确
        is_correct = (true_label == pred_label)
        color = 'green' if is_correct else 'red'
        
        # 设置标题
        true_name = class_names[true_label].replace('_', ' ')
        pred_name = class_names[pred_label].replace('_', ' ')
        
        title = f"True: {true_name}\n"
        title += f"Pred: {pred_name}\n"
        title += f"Conf: {confidence:.1f}%"
        
        ax.set_title(title, fontsize=9, color=color, weight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    return save_path

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', default='outputfiles/Standard/birds/312/False/1.0/best_model_42.pth')
    parser.add_argument('-data_dir', default='/data/project/master/dataset/CUB-200-2011/CUB-200-2011')
    parser.add_argument('-dset', default='birds', type=str)
    parser.add_argument('-batch_size', '-b', default=16, type=int)
    parser.add_argument('-n_attributes', default=312, type=int)
    parser.add_argument('-repeat_concepts', action='store_true')
    parser.add_argument('-rep', default=None, type=float)
    parser.add_argument('-corruption_name', default=None, type=str)
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # 加载模型
    model = load_model(args.model_path)
    
    # 加载类别名称
    class_names = get_class_names(args.data_dir)
    print(f"Loaded {len(class_names)} class names")
    
    # 准备验证数据集
    print("Loading validation dataset...")
    _, validset, _ = selector(args)
    
    print(f"Validation set batches: {len(validset)}")
    
    # 评估模型
    results = evaluate_model(model, validset, class_names)
    
    # 可视化预测结果
    save_path = visualize_predictions(results, class_names)
    
    print(f"\n✓ Validation complete!")
    print(f"✓ Results visualization saved to: {save_path}")

if __name__ == '__main__':
    main()
