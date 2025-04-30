"""
基于步数的训练模块 - 实现ReBerta长文本分类模型的步数训练逻辑
"""

import os
import torch
from tqdm import tqdm
import numpy as np
import logging
import time
import traceback
import csv
from datetime import timedelta

# 导入可视化工具和评估工具
from visualization_utils import save_all_steps_curves
from evaluation_utils import validate_during_training

from config import config
from models.reberta import RecurrentRoBERTa
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils import setup_logging, get_device, create_model, create_optimizer
from utils import create_scheduler, clean_gpu_memory, log_gpu_usage
from utils import error_handler, timer, validate_params, TrainingError,ModelError
from data_utils.dataset import prepare_data_loaders


@error_handler(error_type=TrainingError)
@timer
def train_model_by_steps(model, train_loader, val_loader, optimizer, scheduler, 
                 total_steps, eval_every_steps, device, model_save_path, log_dir=None,
                 gradient_accumulation_steps=1, mixed_precision=False, mixed_precision_type="fp16", compile=False):
    """
    基于步数训练模型
    
    Args:
        model (nn.Module): 模型实例
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        optimizer (Optimizer): 优化器
        scheduler (LRScheduler): 学习率调度器
        total_steps (int): 总训练步数
        eval_every_steps (int): 每多少步进行一次评估
        device (torch.device): 计算设备
        model_save_path (str): 模型保存路径
        log_dir (str, optional): 日志和评估结果保存目录
        gradient_accumulation_steps (int): 梯度累积步数
        mixed_precision (bool): 是否使用混合精度训练
        mixed_precision_type (str): 混合精度类型，可选"fp16"或"bf16"
        compile (bool): 是否使用PyTorch编译器优化模型
        
    Returns:
        tuple: (train_losses, val_losses, train_accuracies, val_accuracies) 训练和验证的损失与准确度列表
    """
    # 激活梯度异常检测模式以找出梯度计算失败的操作
    import torch.autograd
    torch.autograd.set_detect_anomaly(True)
    
    # 如果启用混合精度训练，创建上下文
    from utils import create_mixed_precision_context
    mixed_precision_context = create_mixed_precision_context(mixed_precision, mixed_precision_type)
    
    # 应用PyTorch编译器优化
    if compile and torch.__version__ >= "2.0.0":
        try:
            # 使用torch.compile加速模型
            logger.info("尝试应用PyTorch编译器优化...")
            model = torch.compile(model, mode="reduce-overhead")  # 使用reduce-overhead模式，平衡编译时间和运行性能
            logger.info("PyTorch编译器优化已成功应用")
        except Exception as e:
            logger.warning(f"PyTorch编译器优化失败: {str(e)}")
            logger.warning("继续使用未编译的模型训练")
    
    logger = logging.getLogger(__name__)
    logger.info("已激活梯度异常检测，这将帮助定位反向传播错误")
    
    # 梯度累积相关设置
    if gradient_accumulation_steps > 1:
        logger.info(f"启用梯度累积，每{gradient_accumulation_steps}步更新一次权重")
    
    # 混合精度训练设置
    if mixed_precision:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if mixed_precision and mixed_precision_type == "fp16" else None
        logger.info(f"启用{mixed_precision_type}混合精度训练")
    best_val_loss = float('inf')
    no_improve_steps = 0
    early_stopping_patience = config.get('early_stopping_patience', 5) * eval_every_steps
    
    # 用于存储训练和验证损失以及准确度
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_accuracy = 0
    best_accuracy_model_path = None
    # 训练统计变量
    global_step = 0

    # 自动生成时间戳目录和last_model路径
    if log_dir is None:
        log_dir = os.path.join(config['log_dir'], time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    last_model_path = os.path.join(log_dir, 'last_model.pt')

    # 断点续训机制
    resume_step = 0
    if hasattr(config, 'resume_checkpoint') and config['resume_checkpoint']:
        checkpoint_path = config['resume_checkpoint']
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            resume_step = checkpoint.get('step', 0)
            global_step = resume_step
            logger.info(f"从断点 {checkpoint_path} 恢复训练，起始step={resume_step}")
        else:
            logger.warning(f"resume_checkpoint指定但文件不存在: {checkpoint_path}")

    # CSV文件路径
    csv_file_path = os.path.join(log_dir, "evaluation_metrics.csv")
    if resume_step > 0 and os.path.exists(csv_file_path):
        # 断点续训时读取已有CSV，避免重复写入
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = list(csv.DictReader(csvfile))
            if reader:
                last_row = reader[-1]
                best_accuracy = float(last_row.get('best_accuracy', 0))
                best_accuracy_model_path = last_row.get('best_accuracy_model', None)

    
    train_loss = 0
    train_steps = 0
    train_correct = 0
    train_total = 0
    
    # 记录开始时间
    start_time = time.time()
    
    logger.info(f"开始基于步数的训练，总步数: {total_steps}，每{eval_every_steps}步评估一次")
    model.train()
    
    # 创建无限循环的训练数据迭代器
    train_iterator = iter(train_loader)
    progress_bar = tqdm(range(total_steps), desc="训练中")
    
    for step in progress_bar:
        # 定期清理GPU内存
        if step % 20 == 0 and device.type == 'cuda':
            clean_gpu_memory()
            
        # 定期记录GPU使用情况
        if step % 100 == 0 and device.type == 'cuda':
            log_gpu_usage(f"Step {step}/{total_steps}")
        
        # 获取下一个批次，如果迭代器已耗尽，则重新创建
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        
        # =================== 训练逻辑 ===================
        try:
            # 准备数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            # 仅在梯度累积周期开始时清零梯度
            if step % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
                
            # 使用混合精度上下文进行前向传播
            with mixed_precision_context:
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 如果启用了梯度累积，对损失进行缩放
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
            
            # 计算准确度
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            train_correct += correct
            train_total += labels.size(0)
            batch_acc = correct / labels.size(0)
            
            # 反向传播
            if mixed_precision and mixed_precision_type == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积的最后一步，执行参数更新
            if (step + 1) % gradient_accumulation_steps == 0 or step == total_steps - 1:
                try:
                    # 梯度裁剪，防止梯度爆炸
                    if mixed_precision and mixed_precision_type == "fp16":
                        scaler.unscale_(optimizer)
                    
                    # 将可能抛出异常的梯度裁剪包装在try块内
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    if mixed_precision and mixed_precision_type == "fp16":
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()  # 动态调整学习率
                except RuntimeError as e:
                    # 捕获梯度裁剪可能产生的异常
                    logger.error(f"梯度裁剪或参数更新错误: {str(e)}")
                    
                    # 关键：确保即使在出错时也重置scaler状态
                    if mixed_precision and mixed_precision_type == "fp16" and hasattr(scaler, "_found_inf_per_device"):
                        scaler.update()
                        logger.info("已重置梯度缩放器状态")
                        
                    # 清理GPU内存
                    if device.type == 'cuda':
                        clean_gpu_memory()
            
            # 记录损失
            train_loss += loss.item()
            train_steps += 1
            global_step += 1
            
            # 更新进度条
            curr_acc = train_correct / train_total if train_total > 0 else 0
            curr_loss = train_loss / train_steps
            progress_bar.set_postfix({
                'loss': f"{curr_loss:.4f}",
                'acc': f"{curr_acc:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        except Exception as e:
            logger.error(f"Step {step} 训练出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 如果是内存错误，尝试清理内存
            if "CUDA out of memory" in str(e) and device.type == 'cuda':
                logger.error(f"CUDA内存不足，跳过该批次")
                clean_gpu_memory()
            
            # 如果是台式机生产环境，可能需要保存检查点并继续训练
            if step % 100 == 0:  # 每100步保存一次检查点
                try:
                    checkpoint_path = os.path.join(os.path.dirname(model_save_path), f"emergency_checkpoint_step_{step}.pt")
                    from utils import save_checkpoint
                    label2id = config.get('label2id', None)
                    id2label = config.get('id2label', None)
                    save_checkpoint(checkpoint_path, model, optimizer, scheduler, config, label2id, id2label, extra={'step': step})
                    logger.info(f"已保存紧急检查点到 {checkpoint_path}")
                except Exception as save_error:
                    logger.error(f"检查点保存失败: {str(save_error)}")
            # 继续下一步
            continue
        
        # =================== 周期性评估和保存 ===================
        # 这部分代码不在训练try-except块内，确保即使训练步骤出错，评估仍会执行
        # 进行评估的条件：每eval_every_steps步进行一次评估，或者当前是最后一步
        if global_step % eval_every_steps == 0 or step == total_steps - 1:
                # 计算平均训练损失和准确度
                avg_train_loss = train_loss / train_steps
                train_accuracy = train_correct / train_total if train_total > 0 else 0
                
                # 添加到历史记录
                train_losses.append(avg_train_loss)
                train_accuracies.append(train_accuracy)
                
                # 重置训练统计
                train_loss = 0
                train_steps = 0
                train_correct = 0
                train_total = 0
                
                # 计算耗时
                elapsed_time = time.time() - start_time
                remaining_steps = total_steps - global_step
                steps_per_second = global_step / elapsed_time if elapsed_time > 0 else 0
                estimated_time_left = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                
                # 验证阶段
                logger.info(f"步数 {global_step}/{total_steps} - 已用时: {timedelta(seconds=int(elapsed_time))}, 预计剩余: {timedelta(seconds=int(estimated_time_left))}")
                
                logger.info("执行验证...")
                # 验证前清理内存
                if device.type == 'cuda':
                    clean_gpu_memory()
                    log_gpu_usage(f"Step {global_step} - 验证前")
                
                # 使用评估工具进行验证
                validation_results = validate_during_training(model, val_loader, device, step=global_step)
                
                # 提取验证结果
                avg_val_loss = validation_results.get('loss', float('inf'))
                val_accuracy = validation_results.get('accuracy', 0)
                
                # 添加到列表中用于绘图
                val_losses.append(avg_val_loss)
                val_accuracies.append(val_accuracy)
                
                # 获取更多评估指标
                precision = validation_results.get('precision', 0)
                recall = validation_results.get('recall', 0)
                f1_score = validation_results.get('f1', 0)

                # 保存最佳准确率模型
                is_best_accuracy = val_accuracy > best_accuracy
                if is_best_accuracy:
                    best_accuracy = val_accuracy
                    best_accuracy_model_path = os.path.join(log_dir if log_dir else config['log_dir'], "best_accuracy_model.pt")
                    try:
                        from utils import save_checkpoint
                        label2id = config.get('label2id', None)
                        id2label = config.get('id2label', None)
                        save_checkpoint(best_accuracy_model_path, model, optimizer, scheduler, config, label2id, id2label, extra={
                            'step': global_step,
                            'val_accuracy': val_accuracy,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'train_accuracies': train_accuracies,
                            'val_accuracies': val_accuracies,
                            'history': {
                                'train_losses': train_losses,
                                'val_losses': val_losses,
                                'train_accuracies': train_accuracies,
                                'val_accuracies': val_accuracies,
                                'step': global_step
                            }
                        })
                        logger.info(f"新最佳准确率模型已保存: {best_accuracy_model_path} (准确率: {val_accuracy:.4f})")
                    except Exception as e:
                        logger.error(f"保存最佳准确率模型失败: {str(e)}")

                # 写入CSV
                metrics_row = {
                    'step': global_step,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'best_accuracy': best_accuracy,
                    'best_accuracy_model': best_accuracy_model_path if best_accuracy_model_path else '',
                    'time': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                write_metrics_to_csv(metrics_row)
                
                # 每次评估时保存模型
                eval_model_path = None
                if log_dir is not None:
                    eval_model_path = os.path.join(log_dir, f"eval_model_step_{global_step}.pt")
                    # 创建一个符号链接或副本，用于跟踪最佳模型
                    if val_accuracy > max(val_accuracies) if val_accuracies else 0:
                        best_model_path = os.path.join(log_dir, "best_accuracy_model.pt")
                        logger.info(f"发现准确率最高的模型在步骤 {global_step}")
                else:
                    eval_model_path = os.path.join(os.path.dirname(model_save_path), f"eval_model_step_{global_step}.pt")
                
                # 保存当前评估步骤的模型
                eval_checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                }
                
                try:
                    torch.save(eval_checkpoint, eval_model_path)
                    logger.info(f"评估步骤模型已保存到 {eval_model_path}")
                    
                    # 如果是最佳准确率模型，创建副本
                    if log_dir is not None and val_accuracy >= max(val_accuracies[:-1] + [0]):
                        best_model_path = os.path.join(log_dir, "best_accuracy_model.pt")
                        torch.save(eval_checkpoint, best_model_path)
                        logger.info(f"同时保存为当前最佳准确率模型: {best_model_path} (准确率: {val_accuracy:.4f})")
                except Exception as e:
                    logger.error(f"保存评估步骤模型失败: {str(e)}")
                
                # 确定CSV文件路径
                csv_file_path = None
                if log_dir is not None:
                    csv_file_path = os.path.join(log_dir, "evaluation_metrics.csv")
                else:
                    csv_file_path = os.path.join(os.path.dirname(model_save_path), "evaluation_metrics.csv")
                
                # 检查是否是第一次评估（CSV文件不存在）
                is_first_evaluation = not os.path.exists(csv_file_path)
                
                # 准备CSV行数据
                metrics_row = {
                    'step': global_step,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'model_path': eval_model_path,
                    'time': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 如果是第一次评估，创建CSV文件并写入列标题
                if is_first_evaluation:
                    fieldnames = list(metrics_row.keys())
                    try:
                        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerow(metrics_row)
                        logger.info(f"创建评估指标CSV文件: {csv_file_path}")
                    except Exception as e:
                        logger.error(f"创建CSV文件失败: {str(e)}")
                else:
                    # 读取已有的CSV数据
                    existing_rows = []
                    try:
                        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                            reader = csv.DictReader(csvfile)
                            existing_rows = list(reader)
                    except Exception as e:
                        logger.error(f"读取现有CSV文件失败: {str(e)}")
                    
                    # 添加新行
                    existing_rows.append(metrics_row)
                    
                    # 重写CSV文件
                    try:
                        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                            fieldnames = list(metrics_row.keys())
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            for row in existing_rows:
                                writer.writerow(row)
                        logger.info(f"更新评估指标CSV文件: {csv_file_path}")
                    except Exception as e:
                        logger.error(f"更新CSV文件失败: {str(e)}")
                
                # 验证后清理内存
                if device.type == 'cuda':
                    clean_gpu_memory()
                    log_gpu_usage(f"Step {global_step} - 验证后")
                
                # 打印Loss和准确度
                logger.info(f"步数 {global_step}/{total_steps} - 训练损失: {avg_train_loss:.4f}, 准确度: {train_accuracy:.4f} - 验证损失: {avg_val_loss:.4f}, 准确度: {val_accuracy:.4f}")
                
                # 早停和保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improve_steps = 0
                    
                    # 保存最佳模型
                    save_path = model_save_path
                    
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': global_step,
                        'best_val_loss': best_val_loss,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accuracies': train_accuracies,
                        'val_accuracies': val_accuracies
                    }
                    try:
                        # 保存最佳模型
                        # 保存完整训练历史和配置
                        checkpoint['config'] = config
                        checkpoint['history'] = {
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'train_accuracies': train_accuracies,
                            'val_accuracies': val_accuracies,
                            'step': global_step
                        }
                        from utils import save_checkpoint
                        label2id = config.get('label2id', None)
                        id2label = config.get('id2label', None)
                        save_checkpoint(save_path, model, optimizer, scheduler, config, label2id, id2label, extra=checkpoint)
                        logger.info(f"模型改进！保存到 {save_path}，验证损失: {avg_val_loss:.4f}")
                    except Exception as e:
                        logger.error(f"模型保存失败: {str(e)}")
                        logger.error(traceback.format_exc())
                        # 尝试保存到备用路径
                        try:
                            # 根据是否提供log_dir决定备份路径
                            if log_dir is not None:
                                backup_path = os.path.join(log_dir, f"backup_model_{global_step}.pt")
                            else:
                                backup_path = os.path.join(os.path.dirname(save_path), f"backup_model_{global_step}.pt")
                            logger.info(f"尝试保存到备用路径: {backup_path}")
                            from utils import save_checkpoint
                            label2id = config.get('label2id', None)
                            id2label = config.get('id2label', None)
                            save_checkpoint(backup_path, model, optimizer, scheduler, config, label2id, id2label, extra=checkpoint)
                            logger.info(f"模型成功保存到备用路径: {backup_path}")
                        except Exception as backup_error:
                            logger.error(f"备用保存也失败: {str(backup_error)}")
                else:
                    no_improve_steps += eval_every_steps
                    logger.info(f"无改进: {no_improve_steps}/{early_stopping_patience} 步")
                
                # 检查是否需要早停
                if no_improve_steps >= early_stopping_patience:
                    logger.info(f"早停触发，{early_stopping_patience}步无改进")
                    break
                
                # 切回训练模式
                model.train()
    
    # 训练完成后的可视化
    try:
        # 创建步数轴
        steps_axis = [(i+1) * eval_every_steps for i in range(len(train_losses))]
        
        # 确定曲线图保存目录
        curves_dir = log_dir
        os.makedirs(curves_dir, exist_ok=True)
        
        # 更精细的曲线图文件名
        loss_curve_path = os.path.join(curves_dir, 'loss_curve.png')
        acc_curve_path = os.path.join(curves_dir, 'accuracy_curve.png')
        combined_curve_path = os.path.join(curves_dir, 'training_curves.png')
        
        # 使用可视化工具保存基于步数的曲线
        save_results = save_all_steps_curves(
            train_losses, val_losses, train_accuracies, val_accuracies,
            steps_axis, curves_dir
        )
        
        # 再次检查CSV文件是否存在并记录最终结果
        if os.path.exists(csv_file_path):
            logger.info(f"所有评估指标已保存到CSV文件: {csv_file_path}")
            logger.info(f"可以使用此CSV文件筛选最佳模型")
        
        for curve_type, success in save_results.items():
            if success:
                logger.info(f"{curve_type.capitalize()}曲线保存成功")
            else:
                logger.warning(f"{curve_type.capitalize()}曲线保存失败")
        
        # 保存最后模型
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'config': config,
                'history': {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accuracies': train_accuracies,
                    'val_accuracies': val_accuracies,
                    'step': global_step
                }
            }, last_model_path)
            logger.info(f"最后模型已保存到: {last_model_path}")
        except Exception as e:
            logger.error(f"保存最后模型失败: {str(e)}")
        
    except Exception as e:
        logger.error(f"可视化失败: {str(e)}")
        logger.error(traceback.format_exc())
    
    # 训练结束前清理 GPU 内存
    if device.type == 'cuda':
        clean_gpu_memory()
    
    # 计算总训练时间
    total_time = time.time() - start_time
    logger.info(f"训练完成！总耗时: {timedelta(seconds=int(total_time))}")
    
    # 输出总结信息
    logger.info("="*50)
    logger.info("训练会话总结")
    logger.info(f"- 共运行了 {global_step} 步训练")
    logger.info(f"- 最终最佳验证准确率: {max(val_accuracies) if val_accuracies else 0:.4f}")
    logger.info(f"- 所有评估指标已保存到CSV")
    logger.info(f"- 所有模型文件已保存到目录: {log_dir}")
    logger.info(f"- 最后模型路径: {last_model_path}")
    logger.info("="*50)
    
    return train_losses, val_losses, train_accuracies, val_accuracies





@error_handler(error_type=TrainingError)
@validate_params(total_steps=int, eval_every_steps=int)
def train_steps(total_steps, eval_every_steps, window_size=None, num_repeats=None, max_windows=None,
             batch_size=None, learning_rate=None, log_dir=None, early_stopping_patience=None,
             dropout=None, gradient_accumulation_steps=None, mixed_precision=None, mixed_precision_type=None,
             lr_scheduler=None, lr_final_factor=None, lr_step_size=None, lr_gamma=None, compile=None):
    """
    执行基于步数的训练
    
    Args:
        total_steps (int): 总训练步数
        eval_every_steps (int): 每多少步评估一次
        window_size (int, optional): 滑动窗口大小
        num_repeats (int, optional): CLS向量循环次数
        max_windows (int, optional): 最大窗口数量
        batch_size (int, optional): 批次大小
        learning_rate (float, optional): 学习率
        log_dir (str, optional): 日志和评估结果保存目录
        early_stopping_patience (int, optional): 早停耐心
        dropout (float, optional): dropout率
        gradient_accumulation_steps (int, optional): 梯度累积步数
        mixed_precision (bool, optional): 是否使用混合精度训练
        mixed_precision_type (str, optional): 混合精度类型，可选"fp16"或"bf16"
        lr_scheduler (str, optional): 学习率调度器类型
        lr_final_factor (float, optional): 学习率最终衰减因子
        lr_step_size (int, optional): 学习率衰减步长
        lr_gamma (float, optional): 学习率衰减率
        compile (bool, optional): 是否使用PyTorch编译器优化模型
    """
    # 设置日志
    # 优先使用传入的log_dir，如果没有则使用config中的路径
    log_directory = log_dir if log_dir is not None else config['log_dir']
    setup_logging(log_dir=log_directory, filename="training_steps.log")
    logger = logging.getLogger(__name__)
    
    logger.info(f"所有训练相关文件将保存在: {log_directory}")
    
    # 设置随机种子，保证可重复性
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 如果传入了参数，覆盖默认配置
    if window_size is not None:
        config['window_size'] = window_size
    if num_repeats is not None:
        config['num_repeats'] = num_repeats
    if max_windows is not None:
        config['max_windows'] = max_windows
    if batch_size is not None:
        config['batch_size'] = batch_size
    if learning_rate is not None:
        config['learning_rate'] = learning_rate
    if dropout is not None:
        config['dropout'] = dropout
    if gradient_accumulation_steps is not None:
        config['gradient_accumulation_steps'] = gradient_accumulation_steps
    if mixed_precision is not None:
        config['mixed_precision'] = mixed_precision
    if mixed_precision_type is not None:
        config['mixed_precision_type'] = mixed_precision_type
    if lr_scheduler is not None:
        config['lr_scheduler'] = lr_scheduler
    if lr_final_factor is not None:
        config['lr_final_factor'] = lr_final_factor
    if lr_step_size is not None:
        config['lr_step_size'] = lr_step_size
    if lr_gamma is not None:
        config['lr_gamma'] = lr_gamma
    if compile is not None:
        config['compile'] = compile
        
    # 打印完整的配置信息
    logger.info(f"基本配置: window_size={config['window_size']}, num_repeats={config['num_repeats']}, "
               f"max_windows={config['max_windows']}, batch_size={config['batch_size']}, "
               f"learning_rate={config['learning_rate']}, dropout={config.get('dropout', 0.1)}")
    
    logger.info(f"学习率调度器: {config.get('lr_scheduler', 'linear')}, "
               f"lr_final_factor={config.get('lr_final_factor', 0.05)}, "
               f"lr_step_size={config.get('lr_step_size', 2)}, "
               f"lr_gamma={config.get('lr_gamma', 0.5)}")
    
    logger.info(f"高级训练配置: 梯度累积步数={config.get('gradient_accumulation_steps', 1)}, "
               f"混合精度训练={config.get('mixed_precision', False)}, "
               f"混合精度类型={config.get('mixed_precision_type', 'fp16')}")
    
    # 获取计算设备
    device = get_device()
    
    # 准备数据
    train_path = 'data/train'
    val_path = 'data/validation'
    train_loader, val_loader, num_classes = prepare_data_loaders(
        config['batch_size'], 
        config['max_len'], 
        config['model_name'],
        train_path, 
        val_path
    )
    
    # 指定本地预训练模型路径
    local_pretrained_path = os.path.join(os.path.dirname(__file__), "models", "pretrained")
    
    # 创建模型 (增强错误处理)
    try:
        model = create_model(
            num_classes, 
            config['window_size'], 
            config['num_repeats'], 
            config['max_windows'], 
            local_pretrained_path,
            device,
            dropout=config.get('dropout', 0.1)  # 使用配置的dropout值
        )
    except Exception as e:
        logger.error(f"模型创建失败: {str(e)}")
        # 如果在GPU上创建失败，尝试在CPU上创建
        if device.type == 'cuda':
            logger.info("尝试在CPU上创建模型...")
            device = torch.device('cpu')
            try:
                model = create_model(
                    num_classes, 
                    config['window_size'], 
                    config['num_repeats'], 
                    config['max_windows'], 
                    local_pretrained_path,
                    device,
                    dropout=config.get('dropout', 0.1)  # 使用配置的dropout值
                )
                logger.info("在CPU上成功创建模型")
            except Exception as cpu_error:
                logger.error(f"在CPU上创建模型也失败: {str(cpu_error)}")
                raise ModelError("无法创建模型，训练终止") from cpu_error
        else:
            raise ModelError("无法创建模型，训练终止") from e
    
    # 创建优化器 (已增强错误处理)
    optimizer = create_optimizer(
        model, 
        config['learning_rate'], 
        config['weight_decay']
    )
    
    # 创建学习率调度器
    warmup_steps = int(total_steps * config['warmup_ratio'])
    logger.info(f"总训练步数: {total_steps}, warmup步数: {warmup_steps}")
    
    # 使用改进的学习率调度器创建函数，支持多种衰减策略
    scheduler = create_scheduler(
        optimizer,
        warmup_steps,
        total_steps,
        scheduler_type=config.get('lr_scheduler', 'linear'),
        lr_final_factor=config.get('lr_final_factor', 0.05),
        step_size=config.get('lr_step_size', 2),
        gamma=config.get('lr_gamma', 0.5)
    )
    
    # 记录调用参数和配置
    logger.info(f"调用参数: total_steps={total_steps}, eval_every_steps={eval_every_steps}")
    
    # 如果是GPU，记录内存占用
    if device.type == 'cuda':
        log_gpu_usage("训练开始前")
    
    # 训练模型
    logger.info("开始基于步数的训练...")
    
    # 确保使用当前目录来保存模型
    model_save_path = os.path.join(log_directory, "model.pt") if log_directory else config['model_save_path']
    config['model_save_path'] = model_save_path
    
    train_losses, val_losses, train_accuracies, val_accuracies = train_model_by_steps(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        total_steps,
        eval_every_steps,
        device,
        model_save_path,  # 使用上面定义的模型保存路径
        log_dir=log_directory,  # 使用一致的日志目录
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        mixed_precision=config.get('mixed_precision', False),
        mixed_precision_type=config.get('mixed_precision_type', 'fp16'),
        compile=config.get('compile', False)
    )
    
    # 确保所有文件都保存在同一个目录中
    save_dir = log_dir if log_dir is not None else os.path.dirname(config['model_save_path'])
    
    # 先计算最终的准确率，确保它们在保存模型前已经定义
    final_train_acc = train_accuracies[-1] if train_accuracies else 0
    final_val_acc = val_accuracies[-1] if val_accuracies else 0
    
    # 输出最终效果总结
    logger.info(f"训练完成效果汇总:")
    logger.info(f"最终训练准确度: {final_train_acc:.4f}")
    logger.info(f"最终验证准确度: {final_val_acc:.4f}")
    
    # 再次磁盘保存一份模型，确保它被正确写入
    final_model_path = os.path.join(save_dir, "model.pt")
    try:
        logger.info(f"保存最终模型到: {final_model_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'training_steps': total_steps,
            'final_train_accuracy': final_train_acc,
            'final_val_accuracy': final_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }, final_model_path)
        
        # 将状态字典也单独保存一份，以备需要
        state_dict_path = os.path.join(save_dir, "state_dict.pth")
        torch.save(model.state_dict(), state_dict_path)
        
        logger.info(f"模型已保存到 {final_model_path}")
        logger.info(f"模型状态字典已保存到 {state_dict_path}")
    except Exception as e:
        logger.error(f"最终模型保存失败: {str(e)}")
    
    logger.info(f"所有训练相关文件已保存到目录: {save_dir}")
    # 最终结果已在前面输出，这里不需要重复
    
    # 最终清理内存
    if device.type == 'cuda':
        clean_gpu_memory()
    
    return model


if __name__ == "__main__":
    # 独立运行模块示例
    
    # 执行基于步数的训练
    train_steps(
        total_steps=10000, 
        eval_every_steps=500,
        window_size=config.get('window_size'),
        num_repeats=config.get('num_repeats'),
        max_windows=config.get('max_windows'),
        batch_size=config.get('batch_size'),
        learning_rate=config.get('learning_rate')
    )
