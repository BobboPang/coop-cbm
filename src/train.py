import os
import math
import time
from datetime import timedelta
import torch

from src.data.data_sel import *
from src.model.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY
from src.util.config import N_CLASSES, MIN_LR, LR_DECAY_SIZE, OUTPUT_DIR
from analysis import Logger, AverageMeter
from src.util.train_util import run_epoch, run_epoch_simple

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_X_to_C(args):
    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                      n_attributes=args.n_attributes, expand_dim=args.expand_dim, three_class=args.three_class)
    train(model, args)

def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                            num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                                 num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_X_to_C_to_y(args):
    if args.repeat_concepts:
        concepts_repeated = int(args.rep*args.n_attributes)
        args.n_attributes = args.n_attributes + concepts_repeated
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    train(model, args)

def train_X_to_y(args):
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux)
    train(model, args)

def train_X_to_Cy(args):
    if args.repeat_concepts:
        concepts_repeated = int(args.rep*args.n_attributes)
        args.n_attributes = args.n_attributes + concepts_repeated
    graph_col = getattr(args, 'graph_col', False)
    model = ModelXtoCY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                       n_attributes=args.n_attributes, three_class=args.three_class, connect_CY=args.connect_CY,
                       graph_col=graph_col)
    train(model, args)

def train_probe(args):
    from src.model import probe
    probe.run(args)

def test_time_intervention(args):
    from src.eval import tti
    tti.run(args)

def robustness(args):
    from src.data import gen_spurious
    gen_spurious.run(args)

def hyperparameter_optimization(args):
    from src.model import hyperopt
    hyperopt.run(args)

def train(model, args):

    trainset, validset, test_loader = selector(args)
    imbalance = None
    
    dir = os.path.join(OUTPUT_DIR, args.exp, args.dset, str(args.n_attributes), str(args.col), str(args.attr_loss_weight))
    if not os.path.exists(dir):
        os.makedirs(dir)
    logger = Logger(os.path.join(dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()
    model = model.to(device)

    # 多 GPU 支持：使用 DataParallel 将 batch 均分到多卡
    gpu_ids = getattr(args, 'gpu_ids', None)
    if gpu_ids and len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    elif torch.cuda.device_count() > 1 and getattr(args, 'multi_gpu', False):
        print(f"Using DataParallel on all {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    criterion = torch.nn.CrossEntropyLoss()
    # 只有在需要属性预测的实验中才创建属性损失函数
    if args.use_attr and not args.no_img and args.exp not in ['Standard']:
        attr_criterion = [] #separate criterion (loss function) for each attribute
        if args.weighted_loss:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.BCEWithLogitsLoss())
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
        print(f"Created {len(attr_criterion)} attribute criteria for {args.n_attributes} attributes")
    else:
        attr_criterion = None
        if args.exp == 'Standard':
            print("Standard model: skipping attribute criteria creation")
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    if args.ckpt: #retraining
        train_loader = trainset
        val_loader = validset
    else:
        train_loader = trainset
        val_loader = validset
    
    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0
    
    # 用于计算预计剩余时间
    epoch_times = []
    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*80}\n")

    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, args, is_training=True)
        else:
            train_loss_meter, train_acc_meter = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, is_training=True)
 
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
    
        with torch.no_grad():
            if args.no_img:
                val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, args, is_training=False)
            else:
                val_loss_meter, val_acc_meter = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, attr_criterion, args, is_training=False)

        # 计算本 epoch 用时
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # 计算预计剩余时间
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta = str(timedelta(seconds=int(eta_seconds)))
        
        # 计算总用时
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            # DataParallel 包装时保存内部 module，确保加载兼容性
            save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(save_model, os.path.join(dir, 'best_model_%d.pth' % args.seed))
            best_marker = " 🌟 NEW BEST!"
        else:
            best_marker = ""

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        
        # 打印带颜色和格式的训练信息
        print(f"\n{'─'*80}")
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] | Time: {epoch_time:.1f}s | ETA: {eta} | Total: {total_time_str}")
        print(f"{'─'*80}")
        print(f"  Train → Loss: {train_loss_avg:7.4f} | Acc: {train_acc_meter.avg:6.2f}%")
        print(f"  Val   → Loss: {val_loss_avg:7.4f} | Acc: {val_acc_meter.avg:6.2f}%{best_marker}")
        print(f"  Best  → Epoch: {best_val_epoch:3d} | Acc: {best_val_acc:6.2f}%")
        print(f"{'─'*80}")
        
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                'Val loss: %.4f\tVal acc: %.4f\t'
                'Best val epoch: %d\n'
                % (epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg, best_val_epoch)) 
        logger.flush()
        
        if epoch <= stop_epoch:
            scheduler.step() #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else scheduler.get_lr()[0]
            print(f'  Current learning rate: {current_lr:.6f}')

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("\n⚠️  Early stopping: Low accuracy")
            break
        if epoch - best_val_epoch >= 300:
            print("\n⚠️  Early stopping: No improvement for 300 epochs")
            break
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Total time: {total_time_str}")
    print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_val_epoch}")
    print(f"{'='*80}\n")

