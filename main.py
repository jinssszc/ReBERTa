"""
主入口模块 - 项目的入口点，提供模型训练和评估的统一接口
"""

# 标准库导入
import os
import sys
import logging
import traceback
import argparse
import warnings
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union
import json
import time

# 第三方库导入
import torch
import numpy as np
import csv

# 项目模块导入
from config import config
from utils import (get_device, error_handler, clean_gpu_memory, log_gpu_usage,
                  ReBertaError, load_model_and_config, DataError, ModelError, 
                  TrainingError, ResourceError, ConfigError, EvaluationError, create_model)
from train import train
from train_by_steps import train_steps
from data_utils.dataset import prepare_data_loaders, prepare_test_loader
from evaluation_utils import evaluate_model

# 测试模式设置（用于快速测试代码的健壮性）
TEST_MODE = False  # 设置为True时只处理少量样本
TEST_SAMPLES = 10  # 快速测试时处理的样本数量

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='ReBerta模型训练与评估')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'train_steps', 'evaluate', 'train_and_evaluate'],
                        help='运行模式: train(基于epoch), train_steps(基于步数), evaluate, 或 train_and_evaluate')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default=config.get('data_dir', None), help='数据根目录（可选）')
    parser.add_argument('--train_path', type=str, default=config['data_paths']['train'], help='训练数据路径')
    parser.add_argument('--val_path', type=str, default=config['data_paths']['val'], help='验证数据路径')
    parser.add_argument('--test_path', type=str, default=config['data_paths']['test'], help='测试数据路径')
    parser.add_argument('--sample_data', type=int, default=None, help='数据采样数量，用于快速测试')
    parser.add_argument('--batch_size', type=int, default=config['batch_size'], help='批次大小')
    parser.add_argument('--num_workers', type=int, default=config['num_workers'], help='数据加载器的工作进程数')
    
    # 模型相关参数
    parser.add_argument('--pretrained_path', type=str, default=config['pretrained_path'], 
                        help='RoBERTa预训练模型路径，用于初始化ReBerta模型')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='微调后的ReBerta模型.pt文件路径，用于评估或断点续训')
    parser.add_argument('--model_type', type=str, default=config['model_type'], help='模型类型')
    parser.add_argument('--max_len', type=int, default=config['max_len'], help='最大序列长度')
    parser.add_argument('--window_size', type=int, default=config['window_size'], help='滑动窗口大小')
    parser.add_argument('--num_repeats', type=int, default=config['num_repeats'], help='重复次数')
    parser.add_argument('--max_windows', type=int, default=config['max_windows'], help='最大窗口数')
    parser.add_argument('--dropout', type=float, default=config['dropout'], help='Dropout率')
    parser.add_argument('--hidden_dropout_prob', type=float, default=config['hidden_dropout_prob'], help='隐藏层dropout率')
    parser.add_argument('--attention_dropout_prob', type=float, default=config['attention_dropout_prob'], help='注意力dropout率')
    parser.add_argument('--max_grad_norm', type=float, default=config['max_grad_norm'], help='梯度裁剪范数')
    parser.add_argument('--save_total_limit', type=int, default=None, help='最多保留多少个检查点（None为不限制）')
    parser.add_argument('--fp16', action='store_true', help='是否启用FP16混合精度')
    parser.add_argument('--fp16_opt_level', type=str, default='O1', choices=['O0', 'O1', 'O2', 'O3'], help='FP16优化级别')
    parser.add_argument('--num_classes', type=int, default=config['num_classes'], help='分类类别数（默认自动检测）')
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复训练')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--start_epoch', type=int, default=0, help='从第几轮(epoch)开始训练（用于断点续训）')
    
    # 训练相关参数
    parser.add_argument('--lr_final_factor', type=float, default=config.get('lr_final_factor', 0.05), help='余弦调度器最终学习率因子')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=config['learning_rate'], help='学习率')
    parser.add_argument('--weight_decay', type=float, default=config['weight_decay'], help='权重衰减')
    parser.add_argument('--early_stopping_patience', type=int, default=config['early_stopping_patience'], help='早停耐心值')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=config['gradient_accumulation_steps'], help='梯度累积步数')
    
    # 优化器和学习率调度器
    parser.add_argument('--scheduler_type', type=str, default=config['scheduler_type'],
                        choices=['linear', 'cosine', 'constant', 'constant_with_warmup'],
                        help='学习率调度器类型')
    parser.add_argument('--num_warmup_steps', type=int, default=config['num_warmup_steps'], help='预热步数')
    parser.add_argument('--optimizer_type', type=str, default=config['optimizer']['type'],
                        choices=['adamw', 'adam', 'sgd'],
                        help='优化器类型')
    
    # 混合精度配置
    parser.add_argument('--mixed_precision', action='store_true', default=config['mixed_precision'], help='是否使用混合精度训练')
    parser.add_argument('--mixed_precision_type', type=str, default=config['mixed_precision_type'],
                        choices=['fp16', 'bf16'],
                        help='混合精度类型')
    # torch.compile 配置
    parser.add_argument('--compile', action='store_true', default=False, help='是否使用torch.compile')
    
    # 正则化参数
    parser.add_argument('--label_smoothing', type=float, default=config['regularization']['label_smoothing'], help='标签平滑系数')
    parser.add_argument('--gradient_clip_val', type=float, default=config['regularization']['gradient_clip_val'], help='梯度裁剪值')
    
    # 数据增强
    parser.add_argument('--use_augmentation', action='store_true', default=config['augmentation']['enabled'], help='是否启用数据增强')
    parser.add_argument('--random_mask_prob', type=float, default=config['augmentation']['random_mask_prob'], help='随机掩码概率')
    parser.add_argument('--max_mask_tokens', type=int, default=config['augmentation']['max_mask_tokens'], help='最大掩码token数')
    parser.add_argument('--whole_word_mask', action='store_true', default=config['augmentation']['whole_word_mask'], help='是否使用全词掩码')
    
    # 输出和日志
    parser.add_argument('--output_dir', type=str, default=config['output_dir'], help='输出目录')
    parser.add_argument('--cache_dir', type=str, default=config['cache_dir'], help='缓存目录')
    parser.add_argument('--save_steps', type=int, default=config['epochs'], help='多少步保存一次模型（默认等于epochs）')
    parser.add_argument('--log_level', type=str, default=config['log_level'],
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别')
    parser.add_argument('--log_steps', type=int, default=config['log_steps'], help='每多少步记录一次训练状态')
    parser.add_argument('--save_predictions', action='store_true', default=config['save_predictions'], help='是否保存预测结果')
    
    # 资源配置
    parser.add_argument('--gpu_memory_fraction', type=float, default=config['gpu_memory_fraction'], help='GPU显存使用比例')
    parser.add_argument('--device', type=str, default=config['device'], help='计算设备')
    
    # 解析参数
    args = parser.parse_args()
    
    return args

def update_config_with_args(config, args):
    """
    使用命令行参数更新配置字典
    
    Args:
        config (dict): 原始配置字典
        args (argparse.Namespace): 命令行参数
        
    Returns:
        dict: 更新后的配置字典
    """
    # 创建配置副本
    updated_config = config.copy()
    
    # 数据路径更新
    if args.train_path:
        updated_config['data_paths']['train'] = args.train_path
    if args.val_path:
        updated_config['data_paths']['val'] = args.val_path
    if args.test_path:
        updated_config['data_paths']['test'] = args.test_path
        
    # 模型配置更新
    for key in ['pretrained_path', 'model_path', 'model_type', 'max_len', 'window_size', 
                'num_repeats', 'max_windows', 'dropout']:
        if hasattr(args, key) and getattr(args, key) is not None:
            updated_config[key] = getattr(args, key)
            
    # 训练配置更新
    for key in ['batch_size', 'epochs', 'learning_rate', 'weight_decay',
                'early_stopping_patience', 'gradient_accumulation_steps']:
        if hasattr(args, key) and getattr(args, key) is not None:
            updated_config[key] = getattr(args, key)
            
    # 学习率调度器配置
    if args.scheduler_type:
        updated_config['scheduler_type'] = args.scheduler_type
    if args.num_warmup_steps is not None:
        updated_config['num_warmup_steps'] = args.num_warmup_steps
        
    # 优化器配置
    if args.optimizer_type:
        updated_config['optimizer']['type'] = args.optimizer_type
        
    # 混合精度配置
    if args.mixed_precision:
        updated_config['mixed_precision'] = True
    if args.mixed_precision_type:
        updated_config['mixed_precision_type'] = args.mixed_precision_type
        
    # 正则化配置
    if args.label_smoothing is not None:
        updated_config['regularization']['label_smoothing'] = args.label_smoothing
    if args.gradient_clip_val is not None:
        updated_config['regularization']['gradient_clip_val'] = args.gradient_clip_val
        
    # 数据增强配置
    if args.use_augmentation:
        updated_config['augmentation']['enabled'] = True
    if args.random_mask_prob is not None:
        updated_config['augmentation']['random_mask_prob'] = args.random_mask_prob
    if args.max_mask_tokens is not None:
        updated_config['augmentation']['max_mask_tokens'] = args.max_mask_tokens
    if args.whole_word_mask:
        updated_config['augmentation']['whole_word_mask'] = True
        
    # 资源配置
    if args.num_workers is not None:
        updated_config['num_workers'] = args.num_workers
    if args.gpu_memory_fraction is not None:
        updated_config['gpu_memory_fraction'] = args.gpu_memory_fraction
        
    # 日志配置
    if args.log_level:
        updated_config['log_level'] = args.log_level
    if args.log_steps is not None:
        updated_config['log_steps'] = args.log_steps
    if args.save_predictions:
        updated_config['save_predictions'] = True
        
    return updated_config

def create_training_directory(mode, base_dir=None, timestamp_format="%Y%m%d_%H%M%S"):
    """
    创建基于时间戳的训练结果保存目录
    
    Args:
        mode (str): 运行模式，如'train'、'train_steps'、'epoch'等
        base_dir (str, optional): 基础目录路径，默认为"models/saved"
        timestamp_format (str, optional): 时间戳格式
        
    Returns:
        str: 创建的目录路径
    """
    logger = logging.getLogger(__name__)
    
    # 标准化模式名称
    mode_map = {
        'train': 'epoch',
        'train_steps': 'steps',
        'train_and_evaluate': 'epoch_eval',
        'evaluate': 'eval'
    }
    dir_mode = mode_map.get(mode, mode)
    
    timestamp = datetime.now().strftime(timestamp_format)
    
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "models", "saved")
    
    # 确保基础目录存在
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        logger.debug(f"创建基础目录: {base_dir}")
        
    # 创建本次训练的目录
    train_dir = os.path.join(base_dir, f"{dir_mode}_{timestamp}")
    os.makedirs(train_dir, exist_ok=True)
    
    logger.info(f"创建训练结果目录: {train_dir}")
    return train_dir

@error_handler(error_type=DataError)
def load_data_with_sampling(mode, args, config, device):
    """
    统一的数据加载函数，支持数据取样
    
    Args:
        mode (str): 运行模式 ('train', 'train_steps', 'evaluate', 'train_and_evaluate')
        args (argparse.Namespace): 命令行参数
        config (dict): 配置字典
        device (torch.device): 计算设备
        
    Returns:
        tuple: 根据模式返回不同的数据加载器对象
            - 训练模式: (train_loader, val_loader, num_classes)
            - 评估模式: (test_loader, test_num_classes)
            
    Raises:
        DataError: 数据加载失败时抛出
    """
    logger = logging.getLogger(__name__)
    
    # 计算实际取样大小（用户指定 > 测试模式 > 无限制）
    sample_size = args.sample_data if args.sample_data is not None else (TEST_SAMPLES if TEST_MODE else None)
    
    # 日志取样信息
    sample_source = "用户指定" if args.sample_data else ("测试模式" if TEST_MODE else "无限制")
    sample_log = f"数据取样设置: {sample_size if sample_size else '使用全部数据'} (来源: {sample_source})"
    logger.info(sample_log)
    
    # 训练相关模式
    if mode in ['train', 'train_steps', 'train_and_evaluate']:
        logger.info(f"准备{'基于步数的' if mode == 'train_steps' else ''}训练和验证数据加载器...")
        try:
            # 尝试加载训练和验证数据
            label_encoder, train_loader, val_loader, num_classes = prepare_data_loaders(
                batch_size=args.batch_size,
                max_len=config['max_len'],
                model_name=config['pretrained_path'],
                train_path=config['data_paths']['train'],
                val_path=config['data_paths']['val'],
                train_max_samples=sample_size,  # 只对训练集应用采样限制
                val_max_samples=None            # 验证集使用全部数据
            )
            if train_loader is None or val_loader is None:
                raise DataError("训练或验证数据加载失败")
                
            # 输出数据加载信息
            if sample_size:
                logger.info(f"训练模式采样策略: 训练集最多取样 {sample_size} 个样本进行训练，验证集使用全部数据")
            else:
                logger.info("训练模式采样策略: 使用全部训练和验证数据")
                
            # 返回label_encoder，供后续评估/测试使用
            return label_encoder, train_loader, val_loader, num_classes
            
        except Exception as e:
            if not isinstance(e, DataError):
                logger.error(f"数据加载错误: {str(e)}")
                if device.type == 'cuda':
                    clean_gpu_memory()
                raise DataError(f"数据加载失败: {str(e)}")
            raise
    
    # 评估模式
    elif mode == 'evaluate':
        logger.info("准备测试数据加载器...")
        try:
            # 训练/验证模式下需先加载label_encoder
            label_encoder, _, _, _ = prepare_data_loaders(
                batch_size=args.batch_size,
                max_len=config['max_len'],
                model_name=config['pretrained_path'],
                train_path=config['data_paths']['train'],
                val_path=config['data_paths']['val'],
                train_max_samples=None,
                val_max_samples=None
            )
            test_loader, detected_test_classes = prepare_test_loader(
                batch_size=args.batch_size,
                max_len=config['max_len'],
                model_name=config['pretrained_path'],
                test_path=config['data_paths']['test'],
                label_encoder=label_encoder,
                max_samples=sample_size
            )
            logger.info(f"评估模式采样策略: 使用全部测试数据进行评估")
            return test_loader, detected_test_classes
            
        except Exception as e:
            if not isinstance(e, DataError):
                logger.error(f"测试数据加载错误: {str(e)}")
                if device.type == 'cuda':
                    clean_gpu_memory()
                raise DataError(f"测试数据加载失败: {str(e)}")
            raise
    
    else:
        raise ConfigError(f"不支持的运行模式: {mode}")

@error_handler(error_type=ReBertaError, reraise=False)
def main() -> Optional[Dict[str, Any]]:
    """
    主程序入口
    
    Returns:
        Optional[Dict[str, Any]]: 运行结果数据，如果出错则返回None
    """
    # 解析命令行参数
    args = parse_args()
    
    # 更新配置
    global config
    config = update_config_with_args(config, args)
    
    # 设置日志级别
    logging.basicConfig(
        level=getattr(logging, config['log_level']),
        format=config['log_format']
    )
    logger = logging.getLogger(__name__)
    
    # 记录基本系统信息
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA 版本: {torch.version.cuda}")
        logger.info(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    
    # 参数验证
    try:
        # 基础参数验证
        if args.batch_size <= 0:
            raise ConfigError(f"批次大小必须大于0，当前值: {args.batch_size}")
        if args.epochs <= 0:
            raise ConfigError(f"训练轮数必须大于0，当前值: {args.epochs}")
        if args.learning_rate <= 0:
            raise ConfigError(f"学习率必须大于0，当前值: {args.learning_rate}")
        if args.window_size <= 0:
            raise ConfigError(f"窗口大小必须大于0，当前值: {args.window_size}")
        if args.num_repeats <= 0:
            raise ConfigError(f"CLS向量循环次数必须大于0，当前值: {args.num_repeats}")
        if args.max_windows <= 0:
            raise ConfigError(f"最大窗口数必须大于0，当前值: {args.max_windows}")
            
        # 目录路径验证
        if not os.path.exists(args.data_dir):
            logger.warning(f"数据目录不存在: {args.data_dir}，将尝试创建")
            os.makedirs(args.data_dir, exist_ok=True)
        if not os.path.exists(args.output_dir):
            logger.warning(f"输出目录不存在: {args.output_dir}，将尝试创建")
            os.makedirs(args.output_dir, exist_ok=True)
        if not os.path.exists(args.cache_dir):
            logger.warning(f"缓存目录不存在: {args.cache_dir}，将尝试创建")
            os.makedirs(args.cache_dir, exist_ok=True)
            
        # 训练控制参数验证
        if args.save_steps <= 0:
            raise ConfigError(f"保存步数间隔必须大于0，当前值: {args.save_steps}")
        if args.log_steps <= 0:
            raise ConfigError(f"日志记录步数间隔必须大于0，当前值: {args.log_steps}")
        if args.num_warmup_steps < 0:
            raise ConfigError(f"预热步数不能为负数，当前值: {args.num_warmup_steps}")
        if args.weight_decay < 0:
            raise ConfigError(f"权重衰减系数不能为负数，当前值: {args.weight_decay}")
            
        # dropout参数验证
        if not 0 <= args.hidden_dropout_prob <= 1:
            raise ConfigError(f"隐藏层dropout率必须在[0,1]范围内，当前值: {args.hidden_dropout_prob}")
        if not 0 <= args.attention_dropout_prob <= 1:
            raise ConfigError(f"注意力层dropout率必须在[0,1]范围内，当前值: {args.attention_dropout_prob}")
            
        # 训练优化参数验证
        if args.max_grad_norm <= 0:
            raise ConfigError(f"梯度裁剪范数必须大于0，当前值: {args.max_grad_norm}")
        if args.save_total_limit is not None and args.save_total_limit <= 0:
            raise ConfigError(f"检查点保存数量限制必须大于0，当前值: {args.save_total_limit}")
            
        # 基于步数训练的参数验证
        if args.mode == 'train_steps':
            if args.total_steps <= 0:
                raise ConfigError(f"总步数必须大于0，当前值: {args.total_steps}")
            if args.eval_every <= 0:
                raise ConfigError(f"评估间隔必须大于0，当前值: {args.eval_every}")
            if args.eval_every > args.total_steps:
                logger.warning(f"评估间隔({args.eval_every})大于总步数({args.total_steps})，这意味着只会在训练结束时进行评估")
                
        # 分布式训练参数验证
        if args.fp16 and args.fp16_opt_level not in ['O0', 'O1', 'O2', 'O3']:
            raise ConfigError(f"fp16_opt_level必须是['O0', 'O1', 'O2', 'O3']之一，当前值: {args.fp16_opt_level}")
            
        # 评估模式特定验证
        if args.mode in ['evaluate', 'train_and_evaluate']:
            if args.model_path and not os.path.exists(args.model_path):
                raise ConfigError(f"模型文件不存在: {args.model_path}")
            if args.metric_for_best_model not in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
                raise ConfigError(f"评估指标必须是['loss', 'accuracy', 'precision', 'recall', 'f1']之一，当前值: {args.metric_for_best_model}")
                
    except ConfigError as e:
        logger.error(f"参数验证失败: {str(e)}")
        raise
        
    # 获取计算设备
    try:
        device = get_device(args.device)
        logger.info(f"使用设备: {device}")
        
        # 记录GPU内存使用情况
        if device.type == 'cuda':
            log_gpu_usage("程序启动时")
    except ResourceError as e:
        logger.error(f"设备初始化失败: {str(e)}")
        return None
    
    # 加载数据
    try:
        # 初始化数据加载器变量
        train_loader, val_loader, test_loader = None, None, None
        num_classes = 0
        
        label_encoder = None
        if args.mode in ['train', 'train_steps', 'train_and_evaluate']:
            # 加载训练和验证数据
            logger.info("加载训练和验证数据（一次性加载）...")
            # 路径优先级：命令行参数优先
            config['data_paths']['train'] = args.train_path if args.train_path else config['data_paths']['train']
            config['data_paths']['val'] = args.val_path if hasattr(args, 'val_path') and args.val_path else config['data_paths']['val']
            label_encoder, train_loader, val_loader, detected_num_classes = load_data_with_sampling(args.mode, args, config, device)
            # 命令行优先级：若--num_classes被指定则覆盖自动检测
            num_classes = args.num_classes if args.num_classes is not None else detected_num_classes
            logger.info(f"训练数据类别数: {num_classes}")
        
       # if args.mode == 'train_and_evaluate':
       #2025.4.29 修改：合并 测试训练和训练+测试的逻辑，现在训练后直接测试
            # 复用 label_encoder 进行测试数据加载
            logger.info("加载测试数据（一次性加载，复用 label_encoder）...")
            config['data_paths']['test'] = args.test_path if hasattr(args, 'test_path') and args.test_path else config['data_paths']['test']
            test_loader, detected_test_classes = prepare_test_loader(
                batch_size=config['batch_size'],
                max_len=config['max_len'],
                model_name=config['pretrained_path'],
                test_path=config['data_paths']['test'],
                label_encoder=label_encoder,
                max_samples=args.sample_data if args.sample_data is not None else None
            )
            # 验证测试集类别数与训练集一致
            if detected_test_classes != num_classes:
                raise DataError(f"测试集类别数({detected_test_classes})与训练集类别数({num_classes})不一致")
            logger.info(f"测试数据类别数: {detected_test_classes} (与训练集一致)")
    except DataError as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"发生未预期的错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    # 如果选择基于步数训练，直接调用train_steps函数
    if args.mode == 'train_steps':
        try:
            # 创建基于时间戳的训练结果保存目录
            train_dir = create_training_directory("train_steps")

            # 只设置log_dir，不再拼接model.pt，不设置model_save_path
            config['log_dir'] = train_dir

            # 断点续训参数写入config
            config['resume_checkpoint'] = args.resume_checkpoint

            logger.info(f"开始基于步数的训练，总步数: {args.total_steps}, 每{args.eval_every}步评估一次")
            
            # 调用train_steps并捕获返回值
            model = train_steps(
                total_steps=args.total_steps,
                eval_every_steps=args.eval_every,
                window_size=args.window_size,
                num_repeats=args.num_repeats,
                max_windows=args.max_windows,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                log_dir=train_dir,  # 传入训练目录作为日志目录
                dropout=args.dropout,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                mixed_precision=args.mixed_precision,
                mixed_precision_type=args.mixed_precision_type,
                lr_scheduler=args.lr_scheduler,
                lr_final_factor=args.lr_final_factor,
                lr_step_size=args.lr_step_size,
                lr_gamma=args.lr_gamma,
                compile=args.compile  # 添加缺少的compile参数
            )
            
            # 加载最后一次保存的评估结果
            try:
                metrics_path = os.path.join(train_dir, 'evaluation_metrics.csv')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        rows = list(csv.reader(f))
                        if len(rows) > 1:  # 确保有标题行和至少一行数据
                            last_metrics = rows[-1]
                            headers = rows[0]
                            # 创建最终评估指标字典
                            final_val_metrics = {}
                            for i, header in enumerate(headers):
                                if i > 0:  # 跳过第一列(步数)
                                    try:
                                        final_val_metrics[header] = float(last_metrics[i])
                                    except (ValueError, IndexError):
                                        final_val_metrics[header] = 0.0
                            
                            # 提取关键指标
                            final_train_acc = final_val_metrics.get('train_accuracy', 0.0)
                            final_val_acc = final_val_metrics.get('val_accuracy', 0.0)
                            final_val_loss = final_val_metrics.get('val_loss', 0.0)
                            logger.info(f"最终训练准确率: {final_train_acc:.4f}")
                            logger.info(f"最终验证准确率: {final_val_acc:.4f}")
                            logger.info(f"最终验证损失: {final_val_loss:.4f}")
            except Exception as e:
                logger.error(f"读取评估指标文件失败: {str(e)}")
                final_train_acc = 0.0
                final_val_acc = 0.0
                final_val_loss = 0.0
                logger.warning("找不到评估指标文件，使用默认值")
            
            logger.info("对测试集进行全面评估...")
            test_path = config['data_paths']['test']  # 使用配置中的测试集路径
            
            # 使用已加载的测试数据集，避免重复加载
            logger.info("对测试集进行全面评估（使用已加载的数据集）...")
            
            # 确保test_loader已经加载并有效
            if test_loader is None:
                logger.warning("测试数据加载器未找到，重新加载测试数据...")
                test_loader, _ = prepare_test_loader(
                    batch_size=config['batch_size'],
                    max_len=config['max_len'],
                    model_name=config['pretrained_path'],  # 使用pretrained_path
                    test_path=config['data_paths']['test'],
                    label_encoder=label_encoder,  # 复用训练数据的label_encoder
                    max_samples=None  # 评估时使用全部测试数据
                )
            
            # 获取标签名称
            test_dataset = test_loader.dataset
            label_names = test_dataset.label_encoder.classes_
            
            # 优先加载最佳准确率模型
            best_model_path = os.path.join(train_dir, "best_accuracy_model.pt")
            last_model_path = os.path.join(train_dir, "last_model.pt")
            if os.path.exists(best_model_path):
                final_model_path = best_model_path
                logger.info(f"加载测试集评估模型: {final_model_path}（最佳准确率模型）")
            elif os.path.exists(last_model_path):
                final_model_path = last_model_path
                logger.info(f"未找到best_accuracy_model.pt，使用last_model.pt: {final_model_path}")
            else:
                logger.error("未找到可用的模型权重文件（best_accuracy_model.pt 或 last_model.pt）")
                raise ModelError("未找到可用的模型权重文件")

            try:
                # 加载并验证模型数据
                logger.info(f"加载模型权重文件: {final_model_path}")
                model_data = torch.load(final_model_path, map_location=device, weights_only=False)
                
                # 检查是否缺少标签映射
                if 'label2id' not in model_data or 'id2label' not in model_data:
                    # 尝试从JSON文件加载
                    labels_json_path = final_model_path + "_labels.json"
                    if not os.path.exists(labels_json_path):
                        # 尝试备用文件名
                        labels_json_path = os.path.join(os.path.dirname(final_model_path), "model_labels.json")
                    
                    if os.path.exists(labels_json_path):
                        logger.info(f"模型权重文件中缺少标签映射，从JSON文件加载: {labels_json_path}")
                        with open(labels_json_path, 'r', encoding='utf-8') as f:
                            labels_data = json.load(f)
                            if 'label2id' in labels_data and 'id2label' in labels_data:
                                # 将字符串键转换为整数（如果需要）
                                id2label = labels_data['id2label']
                                # 对于id2label，键必须是字符串，所以需要转换
                                id2label = {int(k): v for k, v in id2label.items()}
                                model_data['id2label'] = id2label
                                model_data['label2id'] = labels_data['label2id']
                                logger.info("成功从JSON文件加载标签映射")
                                
                # 验证模型完整性
                required_keys = ['model_state_dict', 'config', 'label2id', 'id2label']
                missing_keys = [k for k in required_keys if k not in model_data]
                if missing_keys:
                    raise ModelError(f"模型文件缺少必要组件: {', '.join(missing_keys)}")
                
                # 创建模型实例 - 使用预训练路径初始化结构
                logger.info(f"使用预训练路径初始化模型结构: {config['pretrained_path']}")
                model = create_model(
                    num_classes=len(model_data['label2id']),
                    window_size=config['window_size'],
                    num_repeats=config['num_repeats'],
                    max_windows=config['max_windows'],
                    pretrained_path=config['pretrained_path'],  # 使用预训练路径
                    device=device,
                    dropout=config.get('dropout', 0.1)
                )
                
                # 加载微调权重
                logger.info("加载微调权重到模型")
                model.load_state_dict(model_data['model_state_dict'])
                
                # 设置标签映射
                model.label2id = model_data['label2id']
                model.id2label = model_data['id2label']
                
                logger.info("模型加载和验证成功")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                logger.error(traceback.format_exc())
                raise ModelError(f"无法加载模型 {final_model_path}: {str(e)}")
            
            # 评估最终模型
            val_eval_results = evaluate_model(model, test_loader, device, label_names, train_dir)
            # 记录评估结果
            logger.info(f"测试集全面评估结果:")
            logger.info(f"准确率: {val_eval_results['accuracy']:.4f}")
            logger.info(f"精确率: {val_eval_results['precision']:.4f}")
            logger.info(f"召回率: {val_eval_results['recall']:.4f}")
            logger.info(f"F1分数: {val_eval_results['f1']:.4f}")
            
            final_val_precision = val_eval_results['precision']
            final_val_recall = val_eval_results['recall']
            final_val_f1 = val_eval_results['f1']
            
            logger.info("基于步数的训练完成")
            if device.type == 'cuda':
                clean_gpu_memory()
                
            # 返回统一格式的结果
            return {
                'status': 'success', 
                'mode': 'train_steps',
                'train_dir': train_dir,
                'metrics': {
                    'train_accuracy': final_train_acc,
                    'val_accuracy': final_val_acc,
                    'val_loss': final_val_loss,
                    'val_precision': final_val_precision,
                    'val_recall': final_val_recall,
                    'val_f1': final_val_f1
                }
            }
        except Exception as e:
            logger.error(f"基于步数的训练失败: {str(e)}")
            logger.error(traceback.format_exc())
            if device.type == 'cuda':
                clean_gpu_memory()
            if isinstance(e, (ConfigError, ModelError, TrainingError, ResourceError, DataError)):
                raise
            else:
                raise TrainingError(f"基于步数的训练失败: {str(e)}") from e
    
    # 处理其他模式
    logger.info("准备数据集...")
    
    if args.mode in ['train', 'train_and_evaluate']:
        try:
            # 检查是否从检查点恢复训练
            resume_checkpoint = None
            start_epoch = 0
            train_dir = None

            if args.resume:
                if args.resume_checkpoint:
                    # 优先使用命令行指定的恢复检查点
                    resume_checkpoint = args.resume_checkpoint
                    # 使用检查点所在目录作为训练目录
                    train_dir = os.path.dirname(resume_checkpoint)
                    logger.info(f"准备从指定检查点恢复训练: {resume_checkpoint}")
                    logger.info(f"将使用检查点所在目录继续训练: {train_dir}")
                else:
                    # 尝试自动查找最新的epoch模型
                    saved_models_dir = config['output_dir']  # 使用配置的输出目录
                    epoch_models = [f for f in os.listdir(saved_models_dir) if f.startswith("epoch_") and os.path.isdir(os.path.join(saved_models_dir, f))]
                    epoch_models.sort(reverse=True)  # 按时间倒序排列
                    
                    # 查找最近的模型目录中的最后一个epoch模型
                    for epoch_dir in epoch_models:
                        model_dir = os.path.join(saved_models_dir, epoch_dir)
                        epoch_files = [f for f in os.listdir(model_dir) if f.startswith("model_epoch_") and f.endswith(".pt")]
                        if epoch_files:
                            epoch_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]), reverse=True)
                            resume_checkpoint = os.path.join(model_dir, epoch_files[0])
                            train_dir = model_dir  # 使用找到的检查点所在目录
                            logger.info(f"自动找到最新的检查点: {resume_checkpoint}")
                            logger.info(f"将使用检查点所在目录继续训练: {train_dir}")
                            break
                    
                    if not resume_checkpoint:
                        logger.warning("未找到可恢复的检查点，将从头开始训练")
                        # 未找到检查点，创建新目录
                        train_dir = create_training_directory(args.mode)
            elif args.start_epoch > 0:
                # 如果未指定恢复训练但设置了起始epoch
                start_epoch = args.start_epoch
                # 仍然创建新目录
                train_dir = create_training_directory(args.mode)
                logger.info(f"将从epoch {start_epoch}开始训练（不加载检查点）")
            else:
                # 全新训练
                train_dir = create_training_directory(args.mode)

            # 设置日志路径也在同一目录下
            config['log_dir'] = train_dir
            logger.info(f"训练输出目录: {train_dir}")
            
            # 调用train.py中的train函数并捕获返回值，传递已加载的数据加载器
            train_results = train(
                train_loader=train_loader,  # 传递已加载的训练数据加载器
                val_loader=val_loader,      # 传递已加载的验证数据加载器
                num_classes=args.num_classes,    # 传递类别数量（命令行优先）
                device=device,              # 添加设备参数
                train_dir=train_dir,        # 训练目录
                args=args,                  # 命令行参数
                config=config,              # 完整配置
                start_epoch=start_epoch,
                resume_checkpoint=resume_checkpoint
            )
            
            # 解包训练结果
            train_losses, val_losses, train_accuracies, val_accuracies, final_train_acc, final_val_acc = train_results
            
            # 记录训练结果信息
            logger.info(f"训练结果目录: {train_dir}")
            logger.info(f"最终训练准确率: {final_train_acc:.4f}")
            logger.info(f"最终验证准确率: {final_val_acc:.4f}")
            
            logger.info("基于epoch的训练完成")
            if device.type == 'cuda':
                clean_gpu_memory()
            
            # 在训练模式下，进行测试集的全面评估
            try:
                logger.info("对测试集进行全面评估...")
                
                # 使用已加载的测试数据集，避免重复加载
                if test_loader is None:
                    logger.warning("测试数据加载器未找到，重新加载测试数据...")
                    test_loader, _ = prepare_test_loader(
                        batch_size=config['batch_size'],
                        max_len=config['max_len'],
                        model_name=config['pretrained_path'],  # 使用pretrained_path
                        test_path=config['data_paths']['test'],
                        label_encoder=label_encoder,  # 复用训练数据的label_encoder
                        max_samples=None  # 评估时使用全部测试数据
                    )
                
                # 获取标签名称
                test_dataset = test_loader.dataset
                label_names = test_dataset.label_encoder.classes_
                
                # 加载最终模型
                final_model_path = os.path.join(train_dir, "model.pt")
                logger.info(f"加载测试集评估模型: {final_model_path}")
                
                try:
                    # 加载并验证模型
                    model_data = torch.load(final_model_path, map_location=device, weights_only=False)
                    
                    # 检查是否缺少标签映射
                    if 'label2id' not in model_data or 'id2label' not in model_data:
                        # 尝试从JSON文件加载
                        labels_json_path = final_model_path + "_labels.json"
                        if not os.path.exists(labels_json_path):
                            # 尝试备用文件名
                            labels_json_path = os.path.join(os.path.dirname(final_model_path), "model_labels.json")
                        
                        if os.path.exists(labels_json_path):
                            logger.info(f"模型权重文件中缺少标签映射，从JSON文件加载: {labels_json_path}")
                            with open(labels_json_path, 'r', encoding='utf-8') as f:
                                labels_data = json.load(f)
                                if 'label2id' in labels_data and 'id2label' in labels_data:
                                    # 将字符串键转换为整数（如果需要）
                                    id2label = labels_data['id2label']
                                    # 对于id2label，键必须是字符串，所以需要转换
                                    id2label = {int(k): v for k, v in id2label.items()}
                                    model_data['id2label'] = id2label
                                    model_data['label2id'] = labels_data['label2id']
                                    logger.info("成功从JSON文件加载标签映射")
                    
                    # 验证模型完整性
                    required_keys = ['model_state_dict', 'config', 'label2id', 'id2label']
                    missing_keys = [k for k in required_keys if k not in model_data]
                    if missing_keys:
                        raise ModelError(f"模型文件缺少必要组件: {', '.join(missing_keys)}")
                    
                    # 验证配置一致性
                    saved_config = model_data['config']
                    config_mismatch = []
                    
                    # 对模型结构参数进行严格验证 - 这些参数会影响模型功能
                    for key in ['window_size', 'num_repeats', 'max_windows']:
                        if saved_config.get(key) != config.get(key):
                            config_mismatch.append(key)
                    
                    # 对pretrained_path进行特殊处理，只记录不同但不视为错误
                    if 'pretrained_path' in saved_config and saved_config.get('pretrained_path') != config.get('pretrained_path'):
                        logger.warning(f"模型保存时使用的预训练路径({saved_config.get('pretrained_path')})与当前环境中的路径({config.get('pretrained_path')})不同，但这不影响模型功能")
                    
                    # 仅当核心参数不匹配时报错
                    if config_mismatch:
                        raise ModelError(f"模型配置不匹配: {', '.join(config_mismatch)}")
                    
                    # 验证标签映射
                    if not isinstance(model_data['label2id'], dict) or not isinstance(model_data['id2label'], dict):
                        raise ModelError("标签映射格式无效")
                    if len(model_data['label2id']) != len(model_data['id2label']):
                        raise ModelError("标签映射不完整")
                    
                    # 创建模型实例
                    model = create_model(
                        num_classes=len(model_data['label2id']),
                        window_size=config['window_size'],
                        num_repeats=config['num_repeats'],
                        max_windows=config['max_windows'],
                        pretrained_path=config['pretrained_path'],
                        device=device,
                        dropout=config.get('dropout', 0.1)
                    )

                    # 加载权重
                    try:
                        model.load_state_dict(model_data['model_state_dict'])
                    except Exception as e:
                        raise ModelError(f"加载模型权重失败: {str(e)}")
                    
                    # 设置标签映射
                    model.label2id = model_data['label2id']
                    model.id2label = model_data['id2label']
                    
                    logger.info("模型加载和验证成功")
                    
                except Exception as e:
                    logger.error(f"模型加载或验证失败: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise ModelError(f"无法加载或验证模型 {final_model_path}: {str(e)}")
                
                # 评估模型
                from evaluation_utils import evaluate_model
                eval_results = evaluate_model(
                    model=model,
                    data_loader=test_loader,
                    device=device,
                    label_names=label_names,
                    log_dir=train_dir  
                )
                
                # 记录评估结果
                logger.info(f"测试集全面评估结果:")
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"{metric}: {value:.4f}")
                
                # 将评估结果记录到CSV文件
                try:
                    # 确定CSV文件路径
                    test_results_csv = os.path.join(train_dir, "test_results.csv")
                    
                    # 准备CSV行数据
                    # 先提取要记录的指标
                    metrics_to_record = {
                        'accuracy': eval_results.get('accuracy', 0.0),
                        'precision': eval_results.get('precision', 0.0),
                        'recall': eval_results.get('recall', 0.0),
                        'f1': eval_results.get('f1', 0.0),
                        'loss': eval_results.get('loss', 0.0)
                    }
                    
                    # 添加测试时间和模型路径
                    metrics_row = {
                        'model_path': final_model_path,
                        'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        **metrics_to_record
                    }
                    
                    # 添加编号和类别数量
                    metrics_row['num_classes'] = len(model.label2id) if hasattr(model, 'label2id') else 0
                    
                    # 读取已有的CSV数据或创建新文件
                    if os.path.exists(test_results_csv):
                        # 读取现有数据
                        try:
                            with open(test_results_csv, 'r', encoding='utf-8') as csvfile:
                                reader = csv.DictReader(csvfile)
                                existing_rows = list(reader)
                                
                            # 判断是否已有相同模型路径的记录
                            model_exists = False
                            for i, row in enumerate(existing_rows):
                                if row.get('model_path') == final_model_path:
                                    # 替换现有的记录
                                    existing_rows[i] = metrics_row
                                    model_exists = True
                                    break
                                    
                            # 如果不存在，添加新记录
                            if not model_exists:
                                existing_rows.append(metrics_row)
                                
                            # 写入更新后的数据
                            with open(test_results_csv, 'w', newline='', encoding='utf-8') as csvfile:
                                fieldnames = list(metrics_row.keys())
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()
                                for row in existing_rows:
                                    writer.writerow(row)
                            logger.info(f"更新测试结果CSV文件: {test_results_csv}")
                        except Exception as e:
                            logger.error(f"更新测试结果CSV文件出错: {str(e)}")
                    else:
                        # 创建新文件
                        try:
                            with open(test_results_csv, 'w', newline='', encoding='utf-8') as csvfile:
                                fieldnames = list(metrics_row.keys())
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerow(metrics_row)
                            logger.info(f"创建测试结果CSV文件: {test_results_csv}")
                        except Exception as e:
                            logger.error(f"创建测试结果CSV文件出错: {str(e)}")
                except Exception as e:
                    logger.error(f"记录测试结果到CSV文件出错: {str(e)}")
                
                final_val_metrics = eval_results
            except Exception as e:
                logger.error(f"测试集全面评估失败: {str(e)}")
                logger.error(traceback.format_exc())
                final_val_metrics = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
            
            # 如果仅训练模式，返回统一格式的结果
            if args.mode == 'train':
                return {
                    'status': 'success',
                    'mode': 'train',
                    'train_dir': train_dir,
                    'metrics': {
                        'train_accuracy': final_train_acc,
                        'val_accuracy': final_val_acc,
                        'val_precision': final_val_metrics['precision'],
                        'val_recall': final_val_metrics['recall'],
                        'val_f1': final_val_metrics['f1']
                    }
                }
                
        except DataError as e:
            logger.error(f"数据加载错误: {str(e)}")
            if device.type == 'cuda':
                clean_gpu_memory()
            raise
            
        except ModelError as e:
            logger.error(f"模型错误: {str(e)}")
            if device.type == 'cuda':
                clean_gpu_memory()
            raise
            
        except TrainingError as e:
            logger.error(f"训练过程错误: {str(e)}")
            logger.error(traceback.format_exc())
            if device.type == 'cuda':
                clean_gpu_memory()
            raise
            
        except Exception as e:
            logger.error(f"训练过程中发生未知错误: {str(e)}")
            logger.error(traceback.format_exc())
            if device.type == 'cuda':
                clean_gpu_memory()
            raise TrainingError(f"训练过程中发生未知错误: {str(e)}") from e
    
    if args.mode in ['evaluate', 'train_and_evaluate']:
        try:
            # 测试数据路径 (从命令行参数或配置中读取)
            test_path = args.test_path if args.test_path else config.get('test_path', 'data/validation')
            
            # 验证数据路径存在性检查
            if not os.path.exists(test_path):
                raise DataError(f"测试数据路径不存在: {test_path}")
            
            # 准备测试数据加载器
            logger.info(f"加载测试数据集: {test_path}")
            # 使用已加载的测试数据集，避免重复加载
            if test_loader is None:
                logger.warning("测试数据加载器未找到，重新加载测试数据...")
                test_loader, test_num_classes = prepare_test_loader(
                    batch_size=config['batch_size'],
                    max_len=config['max_len'],
                    model_name=config['pretrained_path'],  # 使用pretrained_path
                    test_path=config['data_paths']['test'],
                    max_samples=None  # 评估时使用全部测试数据
                )
            
            # 自动选择最佳模型权重路径（仅在train_and_evaluate模式下自动设置）
            model_file_path = args.model_path
            if args.mode == 'train_and_evaluate':
                # 自动检测新创建训练目录下的所有权重文件
                pt_files = [f for f in os.listdir(train_dir) if f.endswith('.pt')]
                best_acc_path = os.path.join(train_dir, 'best_accuracy_model.pt')
                last_model_path = os.path.join(train_dir, 'last_model.pt')
                default_model_path = os.path.join(train_dir, 'model.pt')
                chosen_path = None
                if 'best_accuracy_model.pt' in pt_files:
                    chosen_path = best_acc_path
                    logger.info(f"自动选择最佳准确率权重: {chosen_path}")
                elif 'last_model.pt' in pt_files:
                    chosen_path = last_model_path
                    logger.info(f"未找到best_accuracy_model.pt，使用last_model.pt: {chosen_path}")
                elif 'model.pt' in pt_files:
                    chosen_path = default_model_path
                    logger.info(f"未找到best_accuracy_model.pt和last_model.pt，使用model.pt: {chosen_path}")
                elif pt_files:
                    # 选择最新修改时间的.pt文件
                    pt_files_full = [os.path.join(train_dir, f) for f in pt_files]
                    pt_files_full.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    chosen_path = pt_files_full[0]
                    logger.info(f"未找到标准命名权重，使用最新权重文件: {chosen_path}")
                else:
                    raise ModelError(f"训练目录 {train_dir} 下未找到任何.pt权重文件")
                model_file_path = chosen_path
            # 其余模式仍然使用args.model_path
            if not os.path.exists(model_file_path):
                raise ModelError(f"模型文件不存在: {model_file_path}")
            
            # 加载预训练模型
            logger.info(f"加载模型从 {model_file_path}")
            
            try:
                # 加载并验证模型
                model_data = torch.load(model_file_path, map_location=device, weights_only=False)
                
                # 验证模型完整性
                required_keys = ['model_state_dict', 'config', 'label2id', 'id2label']
                missing_keys = [k for k in required_keys if k not in model_data]
                if missing_keys:
                    raise ModelError(f"模型文件缺少必要组件: {', '.join(missing_keys)}")
                
                # 验证配置一致性
                saved_config = model_data['config']
                config_mismatch = []
                
                # 对模型结构参数进行严格验证 - 这些参数会影响模型功能
                for key in ['window_size', 'num_repeats', 'max_windows']:
                    if saved_config.get(key) != config.get(key):
                        config_mismatch.append(key)
                
                # 对pretrained_path进行特殊处理，只记录不同但不视为错误
                if 'pretrained_path' in saved_config and saved_config.get('pretrained_path') != config.get('pretrained_path'):
                    logger.warning(f"模型保存时使用的预训练路径({saved_config.get('pretrained_path')})与当前环境中的路径({config.get('pretrained_path')})不同，但这不影响模型功能")
                
                # 仅当核心参数不匹配时报错
                if config_mismatch:
                    raise ModelError(f"模型配置不匹配: {', '.join(config_mismatch)}")
                
                
                # 验证标签映射
                if not isinstance(model_data['label2id'], dict) or not isinstance(model_data['id2label'], dict):
                    raise ModelError("标签映射格式无效")
                if len(model_data['label2id']) != len(model_data['id2label']):
                    raise ModelError("标签映射不完整")
                
                # 创建模型实例
                model = create_model(
                    num_classes=len(model_data['label2id']),
                    window_size=config['window_size'],
                    num_repeats=config['num_repeats'],
                    max_windows=config['max_windows'],
                    pretrained_path=config['pretrained_path'],
                    device=device,
                    dropout=config.get('dropout', 0.1)
                )
                
                # 加载权重
                try:
                    model.load_state_dict(model_data['model_state_dict'])
                except Exception as e:
                    raise ModelError(f"加载模型权重失败: {str(e)}")
                
                # 设置标签映射
                model.label2id = model_data['label2id']
                model.id2label = model_data['id2label']
                
                logger.info("模型加载和验证成功")
                
            except Exception as e:
                logger.error(f"模型加载或验证失败: {str(e)}")
                logger.error(traceback.format_exc())
                raise ModelError(f"无法加载或验证模型 {model_file_path}: {str(e)}")
            
            # 评估模型
            # 根据函数定义调整参数
            eval_results = evaluate_model(
                model=model,
                data_loader=test_loader,
                device=device,
                label_names=label_names,
                log_dir=config['log_dir']  # 修改参数名以匹配函数定义
            )
            
            # 记录评估结果
            logger.info(f"评估结果:")
            for metric, value in eval_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric}: {value:.4f}")
            
            # 返回评估结果
            if args.mode == 'evaluate':
                return {
                    'status': 'success',
                    'mode': 'evaluate',
                    'metrics': {
                        'accuracy': eval_results['accuracy'],
                        'precision': eval_results['precision'],
                        'recall': eval_results['recall'],
                        'f1': eval_results['f1']
                    }
                }
            elif args.mode == 'train_and_evaluate':
                return {
                    'status': 'success',
                    'mode': 'train_and_evaluate',
                    'train_dir': train_dir,
                    'metrics': {
                        'train': {
                            'accuracy': final_train_acc
                        },
                        'val': {
                            'accuracy': final_val_acc,
                            'precision': final_val_metrics['precision'],
                            'recall': final_val_metrics['recall'],
                            'f1': final_val_metrics['f1']
                        },
                        'test': {
                            'accuracy': eval_results['accuracy'],
                            'precision': eval_results['precision'],
                            'recall': eval_results['recall'],
                            'f1': eval_results['f1']
                        }
                    }
                }
            
        except DataError as e:
            logger.error(f"数据加载错误: {str(e)}")
            if device.type == 'cuda':
                clean_gpu_memory()
            raise
            
        except ModelError as e:
            logger.error(f"模型错误: {str(e)}")
            if device.type == 'cuda':
                clean_gpu_memory()
            raise
            
        except Exception as e:
            logger.error(f"评估过程中发生未知错误: {str(e)}")
            logger.error(traceback.format_exc())
            if device.type == 'cuda':
                clean_gpu_memory()
            raise

    if args.mode == 'evaluate':
        try:
            logger.info("加载模型和配置（保证标签一致性）...")
            model, config_ckpt, label2id, id2label = load_model_and_config(args.model_path, device)
            if label2id is None or id2label is None:
                logger.error("未能在模型权重中找到label2id/id2label映射，无法保证标签一致性！")
                raise ValueError("模型权重缺少label2id/id2label，请检查训练和保存流程。")
            num_classes = len(label2id)
            label_names = [id2label[str(i)] if str(i) in id2label else id2label[i] for i in range(num_classes)]
            logger.info(f"模型支持的类别数: {num_classes}")
            logger.info(f"标签类别: {', '.join(label_names)}")
            
            # 加载测试数据时使用模型的label2id
            config['data_paths']['test'] = args.test_path if hasattr(args, 'test_path') and args.test_path else config['data_paths']['test']
            test_loader, detected_test_classes = prepare_test_loader(
                batch_size=args.batch_size,
                max_len=config['max_len'],
                model_name=config['pretrained_path'],  # 使用pretrained_path
                test_path=config['data_paths']['test'],
                label2id=label2id,
                max_samples=args.sample_data if hasattr(args, 'sample_data') and args.sample_data is not None else None
            )
            # 验证测试集类别数与模型类别数一致
            if detected_test_classes != num_classes:
                raise DataError(f"测试集类别数({detected_test_classes})与模型类别数({num_classes})不一致")
            logger.info(f"测试数据类别数: {detected_test_classes} (与模型一致)")
            
            # 开始评估
            logger.info("开始模型评估...")
            from evaluation_utils import evaluate_model, save_evaluation_results
            eval_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else None
            if eval_dir is None:
                from evaluate import create_evaluation_directory
                eval_dir = create_evaluation_directory(model_dir=args.model_path)
            else:
                if not os.path.exists(eval_dir):
                    os.makedirs(eval_dir, exist_ok=True)
            
            eval_results = evaluate_model(model, test_loader, device, label_names, eval_dir)
            if 'error' not in eval_results:
                save_evaluation_results(eval_results, eval_dir)
                logger.info(f"评估结果摘要:")
                logger.info(f"准确率: {eval_results['accuracy']:.4f}")
                logger.info(f"精确率: {eval_results['precision']:.4f}")
                logger.info(f"召回率: {eval_results['recall']:.4f}")
                logger.info(f"F1分数: {eval_results['f1']:.4f}")
                logger.info(f"详细结果已保存到: {eval_dir}")
                
                # 返回统一格式的结果
                return {
                    'status': 'success',
                    'mode': 'evaluate',
                    'eval_dir': eval_dir,
                    'metrics': {
                        'accuracy': eval_results['accuracy'],
                        'precision': eval_results['precision'],
                        'recall': eval_results['recall'],
                        'f1': eval_results['f1']
                    }
                }
            else:
                logger.error(f"评估过程中发生错误: {eval_results['error']}")
                raise EvaluationError(f"模型评估失败: {eval_results['error']}")
                
        except Exception as e:
            logger.error(f"评估过程失败: {str(e)}")
            logger.error(traceback.format_exc())
            if device.type == 'cuda':
                clean_gpu_memory()
            if isinstance(e, (ConfigError, ModelError, EvaluationError, ResourceError, DataError)):
                raise
            else:
                raise EvaluationError(f"评估过程失败: {str(e)}") from e


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"程序执行出错: {str(e)}")
        print(traceback.format_exc())
        print(f"{'='*50}\n")
