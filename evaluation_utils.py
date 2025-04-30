"""
评估工具模块 - 提供模型评估功能和指标计算
"""

import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from utils import error_handler, validate_params, clean_gpu_memory, log_gpu_usage
from visualization_utils import save_confusion_matrix


@error_handler(reraise=False)
@validate_params()
def evaluate_batch(model: torch.nn.Module, 
                  batch: Dict[str, torch.Tensor], 
                  device: torch.device) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    评估单个批次的数据
    
    Args:
        model: 模型
        batch: 包含输入数据的字典
        device: 计算设备
        
    Returns:
        Tuple[torch.Tensor, List[int], List[int]]: (损失值, 真实标签列表, 预测标签列表)
    """
    logger = logging.getLogger(__name__)
    
    # 提取批次数据
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch.get('token_type_ids', None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    labels = batch['labels'].to(device)
    
    try:
        # 设置模型为评估模式
        model.eval()
        
        # 无梯度计算
        with torch.no_grad():
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            # 兼容处理不同的模型输出格式
            if isinstance(outputs, tuple):
                # 处理元组格式 (loss, logits)
                loss = outputs[0]
                logits = outputs[1]
                logger.debug("模型返回元组格式的输出")
            else:
                # 处理对象格式 outputs.loss, outputs.logits
                loss = outputs.loss
                logits = outputs.logits
                logger.debug("模型返回对象格式的输出")
            
            # 获取预测结果
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            true_labels = labels.cpu().numpy().tolist()
            
            return loss, true_labels, preds
            
    except Exception as e:
        logger.error(f"批次评估时出错: {str(e)}")
        # 返回默认值以防错误
        return torch.tensor(0.0), [], []


@error_handler(reraise=False)
@validate_params()
def evaluate_model(model: torch.nn.Module, 
                  data_loader: DataLoader, 
                  device: torch.device,
                  label_names: Optional[List[str]] = None,
                  log_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    评估模型性能
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 计算设备
        label_names: 标签名称列表（用于混淆矩阵可视化）
        log_dir: 日志目录（用于保存混淆矩阵）
        
    Returns:
        Dict[str, Any]: 包含评估指标的字典
    """
    logger = logging.getLogger(__name__)
    logger.info("开始模型评估...")
    
    # 初始化指标
    all_losses = []
    all_preds = []
    all_labels = []
    
    # 记录初始GPU内存使用情况
    if device.type == 'cuda':
        log_gpu_usage("评估开始前")
    
    # 自定义tqdm配置，避免与日志冲突
    from tqdm import tqdm
    from io import StringIO
    import sys
    
    # 替换标准输出，避免与日志交错
    original_stdout = sys.stdout
    tqdm_buffer = StringIO()
    sys.stdout = tqdm_buffer
    
    try:
        # 评估过程中不需要计算梯度
        with torch.no_grad():
            # 使用tqdm显示进度，但将输出重定向
            progress_bar = tqdm(data_loader, 
                               desc="评估中", 
                               file=sys.stdout, 
                               position=0, 
                               leave=True, 
                               ncols=80)
            
            for batch in progress_bar:
                # 评估当前批次
                loss, batch_labels, batch_preds = evaluate_batch(model, batch, device)
                
                # 收集结果
                all_losses.append(loss.item())
                all_labels.extend(batch_labels)
                all_preds.extend(batch_preds)
                
                # 更新进度条
                progress_bar.set_description(f"评估中 Loss: {loss.item():.4f}")
        
        # 还原标准输出
        sys.stdout = original_stdout
        
        # 计算平均损失
        avg_loss = np.mean(all_losses)
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        # 记录评估结果
        log_evaluation_results(accuracy, precision, recall, f1, avg_loss)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        
        # 如果提供了日志目录，则保存混淆矩阵
        if log_dir is not None:
            # 确保日志目录存在
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # 保存混淆矩阵
            confusion_matrix_path = os.path.join(log_dir, 'confusion_matrix.png')
            save_success = save_confusion_matrix(
                all_labels, all_preds, confusion_matrix_path, label_names, "Confusion Matrix"
            )
            
            if save_success:
                logger.info(f"混淆矩阵已保存到: {confusion_matrix_path}")
                results['confusion_matrix_path'] = confusion_matrix_path
            else:
                logger.warning("混淆矩阵保存失败")
                results['confusion_matrix_path'] = None
                
        # 记录GPU内存使用情况
        if device.type == 'cuda':
            log_gpu_usage("评估结束后")
            clean_gpu_memory()
                
        return results
        
    except Exception as e:
        logger.error(f"评估过程中出错: {str(e)}")
        # 返回包含错误信息的结果
        return {
            'error': str(e),
            'loss': float('nan'),
            'accuracy': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan')
        }
    finally:
        # 确保在所有情况下都恢复标准输出
        sys.stdout = original_stdout


def log_evaluation_results(accuracy, precision, recall, f1, loss=None):
    """
    记录评估结果
    
    Args:
        accuracy: 准确率
        precision: 精确率
        recall: 召回率
        f1: F1值
        loss: 损失值
    """
    logger = logging.getLogger(__name__)
    logger.info("-" * 50)
    logger.info("评估结果:")
    if loss is not None:
        logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("-" * 50)


@error_handler(reraise=False)
@validate_params()
def validate_during_training(model: torch.nn.Module, 
                           val_loader: DataLoader, 
                           device: torch.device,
                           epoch: Optional[int] = None,
                           step: Optional[int] = None) -> Dict[str, float]:
    """
    在训练期间进行验证
    
    Args:
        model: 要验证的模型
        val_loader: 验证数据加载器
        device: 计算设备
        epoch: 当前训练轮次（用于日志记录）
        step: 当前训练步数（用于日志记录）
        
    Returns:
        Dict[str, float]: 包含验证指标的字典
    """
    logger = logging.getLogger(__name__)
    
    # 记录验证开始
    if epoch is not None:
        logger.info(f"开始第 {epoch+1} 轮验证...")
    elif step is not None:
        logger.info(f"步数 {step} 处开始验证...")
    else:
        logger.info("开始验证...")
    
    # 执行验证
    results = evaluate_model(model, val_loader, device)
    
    # 记录验证完成
    if epoch is not None:
        logger.info(f"第 {epoch+1} 轮验证完成")
    elif step is not None:
        logger.info(f"步数 {step} 处验证完成")
    else:
        logger.info("验证完成")
    
    return results


@error_handler(reraise=False)
@validate_params()
def predict(model: torch.nn.Module, 
           data_loader: DataLoader, 
           device: torch.device) -> List[int]:
    """
    使用模型进行预测
    
    Args:
        model: 用于预测的模型
        data_loader: 数据加载器
        device: 计算设备
        
    Returns:
        List[int]: 预测标签列表
    """
    logger = logging.getLogger(__name__)
    logger.info("开始预测...")
    
    # 设置模型为评估模式
    model.eval()
    
    # 存储所有预测结果
    all_preds = []
    
    try:
        # 无梯度计算
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                # 提取批次数据
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                
                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # 获取预测结果
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)
        
        logger.info(f"预测完成，共 {len(all_preds)} 个样本")
        return all_preds
        
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        return []
