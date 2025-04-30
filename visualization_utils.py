"""
可视化工具模块 - 提供绘图和图表保存功能
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from typing import List, Optional, Dict, Union, Tuple
from sklearn.metrics import confusion_matrix

from utils import error_handler


@error_handler(reraise=False)
def save_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: str, 
                         label_names: Optional[List[str]] = None, 
                         title: str = "Confusion Matrix") -> bool:
    """
    绘制并保存混淆矩阵
    
    Args:
        y_true (List[int]): 真实标签列表
        y_pred (List[int]): 预测标签列表
        save_path (str): 保存路径
        label_names (List[str], optional): 标签名称列表
        title (str): 图表标题
        
    Returns:
        bool: 是否成功保存
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        if label_names is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=label_names, yticklabels=label_names)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        # 保存图表
        plt.savefig(save_path)
        plt.close()  # 确保图形关闭，避免内存泄漏
        
        logger.info(f"混淆矩阵已保存至 {save_path}")
        return True
    except Exception as e:
        logger.error(f"保存混淆矩阵失败: {str(e)}")
        plt.close()  # 确保发生错误时也关闭图形
        return False


@error_handler(reraise=False)
def save_learning_curves(train_values: List[float], val_values: List[float], save_path: str,
                        x_values: Optional[List[int]] = None, 
                        title: str = "Learning Curves", 
                        y_label: str = "Value",
                        x_label: str = "Epoch") -> bool:
    """
    保存学习曲线（如损失曲线、准确率曲线等）
    
    Args:
        train_values (List[float]): 训练值列表
        val_values (List[float]): 验证值列表
        save_path (str): 保存路径
        x_values (List[int], optional): X轴值列表，如果为None则使用索引
        title (str): 图表标题
        y_label (str): Y轴标签
        x_label (str): X轴标签
        
    Returns:
        bool: 是否成功保存
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建X轴值
        if x_values is None:
            x_values = list(range(1, len(train_values) + 1))
        
        # 绘制曲线
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, train_values, label=f"Train {y_label}")
        plt.plot(x_values, val_values, label=f"Validation {y_label}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        
        # 保存图表
        plt.savefig(save_path)
        plt.close()  # 确保图形关闭，避免内存泄漏
        
        logger.info(f"学习曲线已保存至 {save_path}")
        return True
    except Exception as e:
        logger.error(f"保存学习曲线失败: {str(e)}")
        plt.close()  # 确保发生错误时也关闭图形
        return False


@error_handler(reraise=False)
def save_loss_curve(train_losses: List[float], val_losses: List[float], save_path: str,
                   x_values: Optional[List[int]] = None, title: str = "Training and Validation Loss") -> bool:
    """
    保存损失曲线
    
    Args:
        train_losses (List[float]): 训练损失列表
        val_losses (List[float]): 验证损失列表
        save_path (str): 保存路径
        x_values (List[int], optional): X轴值列表，如果为None则使用索引
        title (str): 图表标题
        
    Returns:
        bool: 是否成功保存
    """
    return save_learning_curves(
        train_losses, val_losses, save_path, x_values, title, "Loss", "Epoch"
    )


@error_handler(reraise=False)
def save_accuracy_curve(train_accuracies: List[float], val_accuracies: List[float], save_path: str,
                       x_values: Optional[List[int]] = None, title: str = "Training and Validation Accuracy") -> bool:
    """
    保存准确率曲线
    
    Args:
        train_accuracies (List[float]): 训练准确率列表
        val_accuracies (List[float]): 验证准确率列表
        save_path (str): 保存路径
        x_values (List[int], optional): X轴值列表，如果为None则使用索引
        title (str): 图表标题
        
    Returns:
        bool: 是否成功保存
    """
    return save_learning_curves(
        train_accuracies, val_accuracies, save_path, x_values, title, "Accuracy", "Epoch"
    )


@error_handler(reraise=False)
def save_steps_curves(train_values: List[float], val_values: List[float], save_path: str,
                     steps: List[int], y_label: str = "Value", title: str = "Training by Steps") -> bool:
    """
    保存基于步数的学习曲线
    
    Args:
        train_values (List[float]): 训练值列表
        val_values (List[float]): 验证值列表
        save_path (str): 保存路径
        steps (List[int]): 步数列表
        y_label (str): Y轴标签
        title (str): 图表标题
        
    Returns:
        bool: 是否成功保存
    """
    return save_learning_curves(
        train_values, val_values, save_path, steps, title, y_label, "Steps"
    )


def save_all_training_curves(train_losses: List[float], val_losses: List[float], 
                             train_accuracies: List[float], val_accuracies: List[float], 
                             log_dir: str, prefix: str = "", epoch_number: int = None) -> Dict[str, bool]:
    """
    一次性保存所有训练相关曲线
    
    Args:
        train_losses (List[float]): 训练损失列表
        val_losses (List[float]): 验证损失列表
        train_accuracies (List[float]): 训练准确率列表
        val_accuracies (List[float]): 验证准确率列表
        log_dir (str): 日志目录
        prefix (str): 文件名前缀
        epoch_number (int, optional): 当前epoch编号，用于生成特定epoch的曲线文件名
        
    Returns:
        Dict[str, bool]: 每个图表保存是否成功的字典
    """
    results = {}
    
    # 确保目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建历史记录目录
    history_dir = os.path.join(log_dir, "training_history")
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    
    # 生成文件名
    filename_prefix = prefix
    if epoch_number is not None:
        # 如果提供了epoch编号，为每x个epoch添加编号到文件名
        # 每3个epoch保存一次带编号的曲线，或者第1个epoch和最后一个epoch
        if epoch_number % 3 == 0 or epoch_number == 1:
            filename_prefix = f"epoch_{epoch_number}_"
    
    # 获取当前进行到的epoch数
    current_epochs = list(range(1, len(train_losses) + 1))
    
    # 保存损失曲线
    loss_path = os.path.join(log_dir, f"{filename_prefix}loss_curve.png")
    results["loss"] = save_loss_curve(train_losses, val_losses, loss_path)
    
    # 保存准确率曲线
    acc_path = os.path.join(log_dir, f"{filename_prefix}accuracy_curve.png")
    results["accuracy"] = save_accuracy_curve(train_accuracies, val_accuracies, acc_path)
    
    # 同时总是保存一份不带epoch编号的最新曲线
    if epoch_number is not None and filename_prefix != "":
        latest_loss_path = os.path.join(log_dir, "latest_loss_curve.png")
        latest_acc_path = os.path.join(log_dir, "latest_accuracy_curve.png")
        save_loss_curve(train_losses, val_losses, latest_loss_path)
        save_accuracy_curve(train_accuracies, val_accuracies, latest_acc_path)
    
    # 保存训练历史累积曲线（记录所有epoch的变化）
    history_loss_path = os.path.join(history_dir, "cumulative_loss_curve.png")
    history_acc_path = os.path.join(history_dir, "cumulative_accuracy_curve.png")
    
    # 为累积曲线图创建更细致的图表
    save_cumulative_curve(
        current_epochs, train_losses, val_losses, 
        history_loss_path, "Loss", "Loss变化趋势 (所有Epoch)"
    )
    
    save_cumulative_curve(
        current_epochs, train_accuracies, val_accuracies, 
        history_acc_path, "Accuracy", "准确度变化趋势 (所有Epoch)"
    )
    
    # 添加累积曲线的结果到返回字典中
    results["cumulative_loss"] = os.path.exists(history_loss_path)
    results["cumulative_accuracy"] = os.path.exists(history_acc_path)
    
    return results


@error_handler(reraise=False)
def save_cumulative_curve(epochs: List[int], train_values: List[float], val_values: List[float], 
                         save_path: str, y_label: str, title: str) -> bool:
    """
    保存包含所有epoch训练过程的累积曲线
    
    Args:
        epochs (List[int]): epoch编号列表
        train_values (List[float]): 训练值列表
        val_values (List[float]): 验证值列表
        save_path (str): 保存路径
        y_label (str): Y轴标签
        title (str): 图表标题
        
    Returns:
        bool: 是否成功保存
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建更精细的图表
        plt.figure(figsize=(12, 8))
        
        # 绘制主曲线
        train_line, = plt.plot(epochs, train_values, 'b-', label=f"Train {y_label}", linewidth=2)
        val_line, = plt.plot(epochs, val_values, 'r-', label=f"Validation {y_label}", linewidth=2)
        
        # 为关键点添加标记
        plt.plot(epochs, train_values, 'bo', alpha=0.6)
        plt.plot(epochs, val_values, 'ro', alpha=0.6)
        
        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 如果数据点多于15个，标记关键节点
        if len(epochs) > 15:
            # 每5个epoch标记一次
            for i in range(0, len(epochs), 5):
                if i < len(epochs):
                    plt.annotate(f"{epochs[i]}", (epochs[i], train_values[i]), 
                                textcoords="offset points", xytext=(0,10), ha='center')
                    plt.annotate(f"{epochs[i]}", (epochs[i], val_values[i]), 
                                textcoords="offset points", xytext=(0,-15), ha='center')
        else:
            # 否则标记每个点
            for i in range(len(epochs)):
                plt.annotate(f"{epochs[i]}", (epochs[i], train_values[i]), 
                            textcoords="offset points", xytext=(0,10), ha='center')
                plt.annotate(f"{epochs[i]}", (epochs[i], val_values[i]), 
                            textcoords="offset points", xytext=(0,-15), ha='center')
        
        # 设置x轴刻度，确保显示所有epoch
        if len(epochs) > 20:
            plt.xticks(range(1, len(epochs)+1, 2))  # 每隔2个epoch显示一次刻度
        else:
            plt.xticks(epochs)
        
        # 添加图例、标签和标题
        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc='best')
        
        # 添加训练统计信息
        min_train = min(train_values)
        max_train = max(train_values)
        min_val = min(val_values)
        max_val = max(val_values)
        
        info_text = (
            f"Train {y_label} - Min: {min_train:.4f}, Max: {max_train:.4f}\n"
            f"Val {y_label} - Min: {min_val:.4f}, Max: {max_val:.4f}\n"
            f"Total Epochs: {len(epochs)}"
        )
        
        # 添加文本框显示统计信息
        plt.figtext(0.02, 0.02, info_text, wrap=True, 
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)  # 更高的dpi提供更好的图像质量
        plt.close()
        
        logger.info(f"累积训练曲线已保存至 {save_path}")
        return True
    except Exception as e:
        logger.error(f"保存累积训练曲线失败: {str(e)}")
        plt.close()  # 确保发生错误时也关闭图形
        return False


def save_all_steps_curves(train_losses: List[float], val_losses: List[float], 
                         train_accuracies: List[float], val_accuracies: List[float], 
                         eval_steps: List[int], log_dir: str) -> Dict[str, bool]:
    """
    一次性保存所有基于步数的训练相关曲线
    
    Args:
        train_losses (List[float]): 训练损失列表
        val_losses (List[float]): 验证损失列表
        train_accuracies (List[float]): 训练准确率列表
        val_accuracies (List[float]): 验证准确率列表
        eval_steps (List[int]): 评估步数列表
        log_dir (str): 日志目录
        
    Returns:
        Dict[str, bool]: 每个图表保存是否成功的字典
    """
    results = {}
    
    # 确保目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 保存损失曲线
    loss_path = os.path.join(log_dir, "steps_loss_curve.png")
    results["loss"] = save_steps_curves(train_losses, val_losses, loss_path, eval_steps, "Loss", 
                                      "Training and Validation Loss")
    
    # 保存准确率曲线
    acc_path = os.path.join(log_dir, "steps_accuracy_curve.png")
    results["accuracy"] = save_steps_curves(train_accuracies, val_accuracies, acc_path, eval_steps, "Accuracy", 
                                         "Training and Validation Accuracy")
    
    return results
