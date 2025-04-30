"""
项目工具模块 - 提供共享功能、工具函数和错误处理机制
"""

import os
import sys
import torch
import logging
import functools
import traceback
import gc
from time import time
from datetime import datetime
from typing import Optional, Callable, Any, Dict, List, Union, Tuple
from torch.utils.data import DataLoader
from torch.optim import AdamW

# 自定义异常类
class ReBertaError(Exception):
    """ReBerta模型基础异常类"""
    pass

class ConfigError(ReBertaError):
    """配置相关错误"""
    pass

class ModelError(ReBertaError):
    """模型相关错误"""
    pass

class DataError(ReBertaError):
    """数据相关错误"""
    pass

class ResourceError(ReBertaError):
    """资源相关错误"""
    pass

class TrainingError(ReBertaError):
    """训练过程错误"""
    pass

class EvaluationError(ReBertaError):
    """评估过程错误"""
    pass

from models.reberta import RecurrentRoBERTa
#from data_utils.dataset import prepare_data_loaders, prepare_test_loader


def save_checkpoint(ckpt_path, model, optimizer, scheduler, config, label2id, id2label, epoch=None, train_losses=None, val_losses=None, train_accuracies=None, val_accuracies=None, extra=None):
    """
    统一保存模型权重、优化器、调度器、配置、标签映射等所有关键信息。
    同时将关键信息保存在权重文件中和单独的配置文件中，确保数据一致性。
    
    Args:
        ckpt_path (str): 保存路径
        model (nn.Module): 模型对象
        optimizer (Optimizer): 优化器
        scheduler (LRScheduler): 学习率调度器
        config (dict): 配置字典
        label2id (dict): 标签到id映射
        id2label (dict): id到标签映射
        epoch (int, optional): 当前epoch
        train_losses/val_losses/train_accuracies/val_accuracies (list, optional): 训练过程记录
        extra (dict, optional): 其他需要保存的信息
    """
    import json
    import yaml
    import os
    
    # 1. 准备标签映射
    label_mapping = {
        'label2id': label2id,
        'id2label': id2label,
        'num_classes': model.num_classes
    }
    
    # 2. 准备模型配置
    model_config = {
        'model_architecture': {
            'num_classes': model.num_classes,
            'window_size': model.window_size,
            'num_repeats': model.num_repeats,
            'max_windows': model.max_windows,
            'dropout': model.dropout.p if hasattr(model.dropout, 'p') else model.dropout,
            'pretrained_path': model.pretrained_path if hasattr(model, 'pretrained_path') else config.get('pretrained_path')
        },
        'tokenizer': {
            'name': model.roberta_config.name_or_path,
            'max_position_embeddings': model.roberta_config.max_position_embeddings
        }
    }
    
    # 3. 准备训练配置
    training_config = {}
    if optimizer:
        training_config.update({
            'learning_rate': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
            'batch_size': config.get('batch_size'),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps'),
            'mixed_precision': config.get('mixed_precision'),
            'mixed_precision_type': config.get('mixed_precision_type'),
        })
    model_config['training'] = training_config
    
    # 4. 准备训练指标和历史
    metrics = {
        'best_val_loss': config.get('best_val_loss'),
        'best_accuracy': config.get('best_accuracy'),
        'precision': config.get('precision'),
        'recall': config.get('recall'),
        'f1_score': config.get('f1_score'),
    }
    training_history = {
        'current_epoch': epoch,
        'metrics': metrics,
        'history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        } if train_losses else {}
    }
    
    # 5. 合并所有配置到roberta_cfg
    roberta_cfg = model.roberta_config.to_dict()
    roberta_cfg.update(model_config['model_architecture'])
    roberta_cfg['tokenizer'] = model_config['tokenizer']
    roberta_cfg['label_mapping'] = label_mapping
    roberta_cfg['training_config'] = training_config
    roberta_cfg['metrics'] = metrics
    roberta_cfg['training_history'] = training_history['history']
    
    # 6. 组装checkpoint（确保包含所有信息）
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': roberta_cfg,
        'epoch': epoch,
        'label_mapping': label_mapping,
        'model_config': model_config,
        'training_config': training_config,
        'metrics': metrics,
        'training_history': training_history,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }
    if extra:
        ckpt.update(extra)

    # 强制转换为int类型
    label2id = {int(k): int(v) for k, v in label2id.items()} if label2id is not None else None
    id2label = {int(k): int(v) for k, v in id2label.items()} if id2label is not None else None

    # 7. 保存checkpoint到磁盘
    torch.save(ckpt, ckpt_path)
    
    # 8. 保存单独的配置文件（使用相同的数据）
    save_dir = os.path.dirname(ckpt_path)
    base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    
    # 8.1 保存标签映射（JSON格式）
    with open(os.path.join(save_dir, f"{base_name}_labels.json"), 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        
    # 8.2 保存模型配置（YAML格式）
    with open(os.path.join(save_dir, f"{base_name}_config.yaml"), 'w', encoding='utf-8') as f:
        yaml.safe_dump(model_config, f, default_flow_style=False, allow_unicode=True)
        
    # 8.3 保存训练历史和指标（JSON格式）
    with open(os.path.join(save_dir, f"{base_name}_training.json"), 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2)
        
    # 8.4 保存完整超参数配置（JSON格式）
    # 合并所有相关配置到一个完整的字典
    full_config = {
        # 基础模型参数
        'model': {
            'architecture': 'RecurrentRoBERTa',
            'pretrained_base': model.roberta_config.name_or_path,
            'num_classes': model.num_classes,
            'window_size': model.window_size,
            'num_repeats': model.num_repeats,
            'max_windows': model.max_windows,
            'dropout': model.dropout.p if hasattr(model.dropout, 'p') else model.dropout,
            'hidden_dropout_prob': config.get('hidden_dropout_prob', 0.1),
            'attention_dropout_prob': config.get('attention_dropout_prob', 0.1),
            'label_smoothing': model.label_smoothing if hasattr(model, 'label_smoothing') else config.get('regularization', {}).get('label_smoothing', 0.0),
            'roberta_config': model.roberta_config.to_dict()
        },
        
        # 训练超参数
        'training': {
            'batch_size': config.get('batch_size'),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
            'max_grad_norm': config.get('max_grad_norm', 1.0),
            'mixed_precision': config.get('mixed_precision', False),
            'mixed_precision_type': config.get('mixed_precision_type', 'float16'),
            'epochs': config.get('epochs', 0),
            'early_stopping_patience': config.get('early_stopping_patience', 0),
        },
        
        # 学习率调度器详情
        'scheduler_details': {
            'type': config.get('scheduler_type'),
            'warmup_steps': config.get('num_warmup_steps'),
            'training_steps': config.get('num_training_steps'),
            'final_factor': config.get('lr_final_factor'),
            'step_size': config.get('lr_step_size'),
            'gamma': config.get('lr_gamma'),
        },
        
        # 优化器配置
        'optimizer': {
            'type': config.get('optimizer', {}).get('type', 'adamw'),
            'learning_rate': optimizer.param_groups[0]['lr'] if optimizer else None,
            'weight_decay': optimizer.param_groups[0]['weight_decay'] if optimizer else None,
            'beta1': config.get('optimizer', {}).get('beta1', 0.9),
            'beta2': config.get('optimizer', {}).get('beta2', 0.999),
            'eps': config.get('optimizer', {}).get('eps', 1e-8),
            'momentum': config.get('optimizer', {}).get('momentum', 0.9),
            'scheduler': config.get('scheduler_type'),  
            'warmup_steps': config.get('num_warmup_steps'),
        },
        
        # 正则化配置
        'regularization': {
            'label_smoothing': config.get('regularization', {}).get('label_smoothing', 0.0),
            'gradient_clip_val': config.get('regularization', {}).get('gradient_clip_val'),
            'weight_decay': config.get('regularization', {}).get('weight_decay', 0.01),
        },
        
        # 数据增强配置
        'augmentation': config.get('augmentation', {
            'enabled': False,
            'random_mask_prob': 0.15,
            'max_mask_tokens': 20,
            'whole_word_mask': True,
        }),
        
        # 数据配置
        'data': {
            'train_path': config.get('data_paths', {}).get('train'),
            'val_path': config.get('data_paths', {}).get('val'),
            'test_path': config.get('data_paths', {}).get('test'),
            'max_len': config.get('max_len'),
            'data_dir': config.get('data_dir'),
        },
        
        # 评估配置
        'evaluation': {
            'metrics': config.get('metrics', []),
            'save_predictions': config.get('save_predictions', False),
            'confusion_matrix': config.get('confusion_matrix', False),
            'classification_report': config.get('classification_report', False),
        },
        
        # 资源配置
        'resources': {
            'num_workers': config.get('num_workers', 4),
            'pin_memory': config.get('pin_memory', True),
            'gpu_memory_fraction': config.get('gpu_memory_fraction', 0.9),
        },
        
        # 日志配置
        'logging': {
            'log_level': config.get('log_level', 'INFO'),
            'log_format': config.get('log_format'),
            'save_logs': config.get('save_logs', True),
            'log_steps': config.get('log_steps', 100),
            'log_dir': config.get('log_dir', 'logs'),
        },
        
        # 路径配置
        'paths': {
            'cache_dir': config.get('cache_dir', 'cache'),
            'output_dir': config.get('output_dir', 'models/saved'),
            'model_save_path': config.get('model_save_path'),
            'base_dir': config.get('base_dir'),
        },
        
        # 标签映射
        'labels': {
            'num_classes': len(label2id) if label2id else 0,
            'label2id': label2id,
            'id2label': id2label,
        },
        
        # 训练结果
        'results': {
            'best_val_loss': config.get('best_val_loss'),
            'best_accuracy': config.get('best_accuracy'),
            'current_epoch': epoch,
        },
        
        # 系统信息
        'system': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(next(model.parameters()).device) if model else None,
            'pytorch_version': torch.__version__,
        }
    }
    
    # 添加额外信息(如果有)
    if extra:
        full_config['extra'] = extra
    
    # 保存完整配置
    with open(os.path.join(save_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(full_config, f, ensure_ascii=False, indent=2)


# 错误处理装饰器
def error_handler(error_type=ReBertaError, reraise=True):
    """
    错误处理装饰器，用于捕获和记录函数执行时的异常
    
    Args:
        error_type (Exception): 要转换为的异常类型
        reraise (bool): 是否重新抛出异常
    
    Returns:
        function: 装饰器函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 记录详细错误信息
                error_msg = f"在执行 {func.__name__} 时出错: {str(e)}"
                logger.error(error_msg)
                logger.error(f"参数: {args}, {kwargs}")
                logger.error(traceback.format_exc())
                
                # 如果是GPU相关错误，尝试清理内存
                if 'CUDA' in str(e) or 'cuda' in str(e) or 'GPU' in str(e):
                    logger.info("检测到GPU错误，尝试清理内存...")
                    clean_gpu_memory()
                
                # 转换为指定类型的异常
                if reraise:
                    if isinstance(e, error_type):
                        raise
                    else:
                        raise error_type(error_msg) from e
                return None
        return wrapper
    return decorator


# 定义计时器装饰器
def timer(func):
    """
    计时器装饰器，用于记录函数的执行时间
    
    Args:
        func (function): 要装饰的函数
    
    Returns:
        function: 装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time()
        logger.info(f"开始执行 {func.__name__}...")
        result = func(*args, **kwargs)
        end_time = time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} 执行完成，耗时: {execution_time:.2f} 秒")
        return result
    return wrapper


# GPU内存管理函数
def clean_gpu_memory():
    """
    清理GPU内存，释放缓存和未使用的引用
    
    Returns:
        bool: 是否成功清理内存
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 执行垃圾回收
        gc.collect()
        
        # 检查是否有CUDA可用
        if torch.cuda.is_available():
            # 清理PyTorch的CUDA缓存
            torch.cuda.empty_cache()
            
            # 获取当前内存使用情况
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
            
            logger.info(f"GPU内存清理完成 - 已分配: {allocated:.2f}MB, 已保留: {reserved:.2f}MB")
            return True
        else:
            logger.warning("CUDA不可用，无法清理GPU内存")
            return False
    except Exception as e:
        logger.error(f"GPU内存清理失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def log_gpu_usage(message=""):
    """
    记录当前GPU使用情况
    
    Args:
        message (str): 额外的日志消息
    """
    logger = logging.getLogger(__name__)
    
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            
            msg = f"GPU使用情况{' - ' + message if message else ''}: "
            msg += f"设备: {device_name}, 已分配: {allocated:.2f}MB, 已保留: {reserved:.2f}MB"
            logger.info(msg)
        else:
            logger.info(f"CUDA不可用{' - ' + message if message else ''}")
    except Exception as e:
        logger.error(f"记录GPU使用情况失败: {str(e)}")


def validate_params(**param_types):
    """
    参数验证装饰器，用于检查函数参数类型和有效性
    
    Args:
        param_types: 参数名称与类型的映射
    
    Returns:
        function: 装饰器函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数参数的实际值
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, expected_type in param_types.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    # 类型检查
                    if expected_type is not None and not isinstance(value, expected_type) and value is not None:
                        raise ConfigError(f"参数 {param_name} 类型错误，期望 {expected_type}\n实际收到: {type(value)}")
                    
                    # 特定类型的参数有效性检查
                    if expected_type == int and value is not None:
                        if (param_name.endswith('_size') or param_name.endswith('_len')) and value <= 0:
                            raise ConfigError(f"参数 {param_name} 必须大于0\n实际收到: {value}")
                    elif expected_type == float and value is not None:
                        if param_name == 'learning_rate' and (value <= 0 or value >= 1):
                            raise ConfigError(f"学习率应在(0, 1)范围内\n实际收到: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


@error_handler(error_type=ConfigError)
def setup_logging(log_dir="logs", filename="training.log"):
    """
    设置日志
    
    Args:
        log_dir (str): 日志目录
        filename (str): 日志文件名
        
    Returns:
        logging.Logger: 日志器对象
        
    Raises:
        ConfigError: 如果日志配置失败
    """
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 检查日志文件路径是否可写
        log_path = os.path.join(log_dir, filename)
        try:
            # 尝试创建文件来检查写权限
            with open(log_path, 'a') as f:
                pass
        except IOError:
            raise ConfigError(f"无法写入日志文件: {log_path}")
        
        # 重置根记录器，避免重复配置
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers[:]:
                root.removeHandler(handler)
        
        # 定义彩色控制台处理器
        class ColoredConsoleHandler(logging.StreamHandler):
            """自定义彩色控制台处理器"""
            
            COLORS = {
                'DEBUG': '\033[94m',     # 蓝色
                'INFO': '\033[92m',      # 绿色
                'WARNING': '\033[93m',   # 黄色
                'ERROR': '\033[91m',     # 红色
                'CRITICAL': '\033[91m\033[1m',  # 红色加粗
                'RESET': '\033[0m'       # 重置颜色
            }
            
            def emit(self, record):
                # 确保在多线程环境中安全输出
                self.acquire()
                try:
                    # 特殊处理测试模式消息
                    if 'TEST_MODE' in record.getMessage() or '测试模式' in record.getMessage():
                        record.levelname = "TEST_" + record.levelname
                    
                    # 为错误消息添加分隔符
                    if record.levelno >= logging.ERROR:
                        msg = record.msg
                        if not msg.startswith('\n==='):
                            record.msg = f"\n{'='*80}\n{msg}\n{'='*80}"
                    
                    # 应用颜色
                    levelname = record.levelname
                    if levelname.startswith('TEST_'):
                        level_color = '\033[44m'  # 蓝色背景
                        pure_level = levelname[5:]  # 去掉TEST_前缀
                        if pure_level in self.COLORS:
                            level_color += self.COLORS[pure_level]
                    elif levelname in self.COLORS:
                        level_color = self.COLORS[levelname]
                    else:
                        level_color = ''
                    
                    # 创建格式化器并应用
                    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    formatted_msg = formatter.format(record)
                    
                    # 添加颜色并输出
                    if level_color:
                        formatted_msg = level_color + formatted_msg + self.COLORS['RESET']
                    
                    # 确保完整的消息作为一个单元输出，避免交错
                    print(formatted_msg)
                finally:
                    self.release()
        
        # 文件处理器 - 无颜色
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        
        # 自定义控制台处理器
        console_handler = ColoredConsoleHandler()
        console_handler.setLevel(logging.INFO)
        
        # 配置根日志记录器
        root.setLevel(logging.INFO)
        root.addHandler(file_handler)
        root.addHandler(console_handler)
        
        # 创建并返回日志器
        logger = logging.getLogger(__name__)
        logger.info("=== ReBerta 日志系统初始化 ===")
        logger.info(f"日志将保存到: {log_path}")
        
        return logger
    except Exception as e:
        if not isinstance(e, ConfigError):
            raise ConfigError(f"配置日志系统失败: {str(e)}") from e
        raise


@error_handler(error_type=ModelError)
@validate_params(num_classes=int, window_size=int, num_repeats=int, max_windows=int)
@timer
def create_model(num_classes, window_size, num_repeats, max_windows, pretrained_path=None, model_path=None, device=None, dropout=0.1, compile=False, compile_mode='reduce-overhead'):
    """
    创建或加载ReBerta模型实例
    
    Args:
        num_classes (int): 分类类别数
        window_size (int): 滑动窗口大小
        num_repeats (int): 迭代次数
        max_windows (int): 最大窗口数
        pretrained_path (str, optional): RoBERTa预训练模型路径，用于初始化ReBerta内部编码器
        model_path (str, optional): 微调后的ReBerta模型权重文件(.pt)，用于加载完整模型
        device (torch.device, optional): 计算设备
        dropout (float, optional): dropout率
        compile (bool, optional): 是否使用torch.compile编译模型(需要PyTorch 2.0+)
        compile_mode (str, optional): 编译模式，默认为'reduce-overhead'
        
    Returns:
        RecurrentRoBERTa: 模型实例
        
    Raises:
        ModelError: 如果模型创建失败
        ConfigError: 如果参数无效
    """
    logger = logging.getLogger(__name__)
    
    # 检查参数有效性
    if num_classes <= 1:
        raise ConfigError(f"分类数必须大于1，当前值: {num_classes}")
    if window_size <= 0:
        raise ConfigError(f"窗口大小必须为正数，当前值: {window_size}")
    if num_repeats <= 0:
        raise ConfigError(f"迭代次数必须为正数，当前值: {num_repeats}")
    if max_windows <= 0:
        raise ConfigError(f"最大窗口数必须为正数，当前值: {max_windows}")
    
    import torch
    import os
    model = None
    
    # 优先使用微调模型路径(如果提供)
    if model_path and os.path.exists(model_path):
        logger.info(f"加载微调模型权重: {model_path}")
        try:
            # 加载并验证模型数据
            model_data = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 验证模型数据的基本结构
            if 'model_state_dict' in model_data:
                # 先使用预训练路径初始化模型
                logger.info(f"使用 RoBERTa 预训练目录初始化模型结构: {pretrained_path}")
                model = RecurrentRoBERTa(
                    num_classes=num_classes,
                    window_size=window_size,
                    num_repeats=num_repeats,
                    max_windows=max_windows,
                    pretrained_path=pretrained_path,
                    dropout=dropout
                )
                # 然后加载微调权重
                model.load_state_dict(model_data['model_state_dict'])
                logger.info("微调模型权重加载成功")
                
                # 恢复标签映射(如果存在)
                if 'label2id' in model_data and 'id2label' in model_data:
                    model.label2id = model_data['label2id']
                    model.id2label = model_data['id2label']
                    logger.info("已恢复模型标签映射")
            else:
                logger.warning(f"模型文件格式不正确，未找到model_state_dict: {model_path}")
        except Exception as e:
            logger.error(f"加载微调模型失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise ModelError(f"无法加载微调模型 {model_path}: {str(e)}")
    # 如果没有提供微调模型，则使用预训练目录初始化
    elif pretrained_path and os.path.exists(pretrained_path):
        # 判断预训练路径是目录还是文件
        if os.path.isdir(pretrained_path):
            logger.info(f"使用 RoBERTa 预训练目录初始化模型: {pretrained_path}")
            model = RecurrentRoBERTa(
                num_classes=num_classes,
                window_size=window_size,
                num_repeats=num_repeats,
                max_windows=max_windows,
                pretrained_path=pretrained_path,
                dropout=dropout
            )
        else:
            logger.warning(f"预训练路径不是有效的目录: {pretrained_path}，将使用默认初始化")
    
    # 如果以上方法都未创建模型，则使用默认初始化
    if model is None:
        logger.info("未提供有效的预训练或微调模型路径，使用默认初始化模型")
        model = RecurrentRoBERTa(
            num_classes=num_classes,
            window_size=window_size,
            num_repeats=num_repeats,
            max_windows=max_windows,
            pretrained_path=None,
            dropout=dropout
        )
    
    # 情况记录
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    # 如果指定了设备，将模型移动到该设备
    if device:
        try:
            logger.info(f"将模型移动到设备: {device}")
            model = model.to(device)
            if device.type == 'cuda':
                log_gpu_usage("模型加载后")
        except Exception as e:
            logger.error(f"模型移动到设备{device}时出错: {str(e)}")
            raise ModelError(f"无法将模型移动到设备 {device}")
    
    # 如果启用了编译并且PyTorch版本支持，应用torch.compile
    if compile:
        try:
            # 检查PyTorch版本是否支持compile
            if hasattr(torch, 'compile'):
                logger.info(f"使用torch.compile编译模型，模式: {compile_mode}")
                model = torch.compile(model, mode=compile_mode)
                logger.info("模型编译成功")
            else:
                logger.warning("当前PyTorch版本不支持torch.compile，跳过编译")
        except Exception as e:
            logger.warning(f"模型编译失败: {str(e)}，使用未编译版本继续")
    
    logger.info("模型创建或加载成功")
    return model


@error_handler(error_type=TrainingError)
@validate_params(learning_rate=float, weight_decay=float)
def create_optimizer(model, learning_rate, weight_decay):
    """
    创建优化器
    
    Args:
        model (nn.Module): 模型实例
        learning_rate (float): 学习率
        weight_decay (float): 权重衰减
        
    Returns:
        AdamW: 优化器实例
        
    Raises:
        TrainingError: 如果优化器创建失败
        ConfigError: 如果参数无效
    """
    logger = logging.getLogger(__name__)
    
    # 检查参数有效性
    if learning_rate <= 0 or learning_rate >= 1:
        logger.warning(f"学习率不在有效范围内，当前值: {learning_rate}，将使用默认值: 2e-5")
        learning_rate = 2e-5  # 设置默认学习率，确保非零
        
    if weight_decay < 0:
        raise ConfigError(f"权重衰减应该非负，当前值: {weight_decay}")
    
    try:
        # 检查模型参数
        if not hasattr(model, 'parameters'):
            raise TrainingError("无效的模型对象，缺少parameters方法")
            
        if not any(p.requires_grad for p in model.parameters()):
            logger.warning("警告: 模型没有可训练参数")
        
        logger.info(f"创建 AdamW 优化器: learning_rate={learning_rate}, weight_decay={weight_decay}")
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        return optimizer
    except Exception as e:
        if not isinstance(e, (TrainingError, ConfigError)):
            raise TrainingError(f"创建优化器失败: {str(e)}") from e
        raise


@error_handler(error_type=TrainingError)
@validate_params(num_warmup_steps=int, num_training_steps=int)
def create_scheduler(optimizer, num_warmup_steps, num_training_steps=None, scheduler_type="linear", lr_final_factor=0.05, step_size=None, gamma=None):
    """
    创建学习率调度器
    
    Args:
        optimizer (Optimizer): 优化器
        num_warmup_steps (int): 预热步数
        num_training_steps (int): 总训练步数
        scheduler_type (str): 调度器类型，可选值: "linear"(默认), "cosine", "step"
        lr_final_factor (float): 最终学习率与初始学习率的比例(用于cosine和linear)
        step_size (int): 学习率阶梯式衰减的步长(用于step)
        gamma (float): 学习率衰减系数(用于step)
        
    Returns:
        LRScheduler: 学习率调度器
        
    Raises:
        TrainingError: 如果调度器创建失败
        ConfigError: 如果参数无效
    """
    logger = logging.getLogger(__name__)
    
    # 检查参数有效性
    if num_warmup_steps < 0:
        raise ConfigError(f"预热步数应该非负，当前值: {num_warmup_steps}")

    # 自动推断 num_training_steps
    if num_training_steps is None:
        # 优先尝试从 optimizer 关联属性自动推断
        num_training_steps = getattr(optimizer, 'num_training_steps', None)
        if num_training_steps is None:
            # 尝试从 optimizer.config 获取
            config = getattr(optimizer, 'config', None)
            train_loader = getattr(optimizer, 'train_loader', None)
            epochs = getattr(optimizer, 'epochs', None)
            if config is not None and 'epochs' in config and 'train_loader' in config:
                try:
                    num_training_steps = len(config['train_loader']) * config['epochs']
                except Exception:
                    num_training_steps = None
            elif train_loader is not None and epochs is not None:
                try:
                    num_training_steps = len(train_loader) * epochs
                except Exception:
                    num_training_steps = None
        if num_training_steps is None:
            raise ConfigError("create_scheduler: 无法自动推断 num_training_steps，请手动传递。")

    
    try:
        # 检查优化器对象
        if not hasattr(optimizer, 'param_groups'):
            raise TrainingError("无效的优化器对象")
        
        # 根据配置选择合适的学习率调度器
        from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
        
        # 获取初始学习率
        initial_lr = optimizer.param_groups[0]['lr']
        logger.info(f"优化器初始学习率: {initial_lr:.10e}")
        
        if scheduler_type.lower() == "linear":
            logger.info(f"创建线性预热调度器: warmup_steps={num_warmup_steps}")
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return 0.1 + 0.9 * float(current_step) / float(max(1, num_warmup_steps))
                return 1.0  # 不再衰减
            scheduler = LambdaLR(optimizer, lr_lambda)
            logger.info(f"线性调度器初始学习率: {scheduler.get_last_lr()[0]:.10e}")
        elif scheduler_type.lower() == "cosine":
            logger.info(f"创建余弦退火预热调度器: warmup_steps={num_warmup_steps}, training_steps={num_training_steps}, lr_final_factor={lr_final_factor}")
            import math
            def cosine_lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return 0.1 + 0.9 * float(current_step) / float(max(1, num_warmup_steps))
                # 余弦退火阶段
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                # 确保余弦值在[lr_final_factor, 1.0]范围内
                return max(
                    lr_final_factor,  # 最低不低于最终因子
                    0.5 * (1.0 + math.cos(math.pi * progress))
                )
            scheduler = LambdaLR(optimizer, cosine_lr_lambda)
            logger.info(f"余弦调度器初始学习率: {scheduler.get_last_lr()[0]:.10e}")
        elif scheduler_type.lower() == "step":
            if step_size is None or gamma is None:
                raise ConfigError("使用step调度器时，必须提供step_size和gamma参数")
            
            logger.info(f"创建阶梯式衰减调度器: step_size={step_size}, gamma={gamma}, 使用单独的预热调度")
            # 首先创建一个预热的调度器
            warmup_scheduler = None
            if num_warmup_steps > 0:
                def lr_lambda(current_step):
                    if current_step < num_warmup_steps:
                        # 预热阶段：从initial_lr * 0.1逐渐增加到initial_lr
                        return 0.1 + 0.9 * float(current_step) / float(max(1, num_warmup_steps))
                    
                    # 线性衰减阶段
                    return max(
                        lr_final_factor,  # 保证不会低于最终因子
                        float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
                    )
                    
                warmup_scheduler = LambdaLR(optimizer, lr_lambda)
                
                # 预热完成后应用StepLR
                for _ in range(num_warmup_steps):
                    warmup_scheduler.step()
                
            # 创建阶梯式衰减调度器
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ConfigError(f"不支持的学习率调度器类型: {scheduler_type}，支持的类型有: linear, cosine, step")
        
        return scheduler
    except Exception as e:
        if not isinstance(e, (TrainingError, ConfigError)):
            raise TrainingError(f"创建学习率调度器失败: {str(e)}") from e
        raise


@error_handler(error_type=ResourceError)
def create_mixed_precision_context(enabled=True, mixed_precision_type="fp16"):
    """
    创建混合精度训练上下文
    
    Args:
        enabled (bool): 是否启用混合精度训练
        mixed_precision_type (str): 混合精度类型，可选值: "fp16"(默认) 或 "bf16"
        
    Returns:
        context: 混合精度训练上下文或空上下文(如果不启用)
        
    Raises:
        ResourceError: 如果创建上下文失败
    """
    logger = logging.getLogger(__name__)
    
    if not enabled:
        logger.info("混合精度训练未启用，使用默认精度")
        # 返回空上下文管理器
        from contextlib import nullcontext
        return nullcontext()
    
    try:
        # 检查是否支持混合精度训练
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，无法启用混合精度训练，将使用默认精度")
            from contextlib import nullcontext
            return nullcontext()
        
        # 简化混合精度上下文创建逻辑，避免版本兼容性问题
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            # 新版PyTorch (≥1.10)
            dtype = torch.float16  # 默认类型
            if mixed_precision_type.lower() == "bf16":
                if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                    logger.info("启用BF16混合精度训练")
                else:
                    logger.warning("当前设备不支持BF16，使用FP16")
            else:
                logger.info("启用FP16混合精度训练")
            
            return torch.amp.autocast(device_type='cuda', dtype=dtype)
        else:
            # 旧版PyTorch
            from torch.cuda.amp import autocast
            logger.info("使用旧版PyTorch API的FP16混合精度训练")
            return autocast()
    except Exception as e:
        logger.error(f"创建混合精度上下文失败: {str(e)}")
        logger.warning("将使用默认精度")
        from contextlib import nullcontext
        return nullcontext()


# prepare_data_loaders 函数已移至data_utils.dataset模块


def load_model_and_config(model_path, device=None):
    """
    通用模型及配置加载函数。
    读取模型权重文件，返回模型、config、label2id和id2label。
    Args:
        model_path (str): 模型权重文件路径
        device (torch.device, optional): 加载到的设备
    Returns:
        model: 加载好的模型对象
        config: 配置字典
        label2id: 标签到id的映射
        id2label: id到标签的映射
        training_state: 训练状态字典，包含优化器、调度器状态等
    Raises:
        ModelError: 如果加载失败
    """
    try:
        # 1. 加载checkpoint
        logger = logging.getLogger(__name__)
        logger.info(f"加载模型权重: {model_path}")
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 2. 提取配置
        config = ckpt['config']
        label2id = config.get('label2id')
        id2label = config.get('id2label')
        
        if not label2id or not id2label:
            raise ModelError("模型权重文件中缺少标签映射")
            
        # 3. 提取训练配置和性能指标
        training_config = config.get('training_config', {})
        metrics = config.get('metrics', {})
        logger.info(f"加载训练配置: {training_config}")
        logger.info(f"加载性能指标: {metrics}")
        
        # 4. 使用保存的配置参数实例化模型
        from models.reberta import RecurrentRoBERTa
        model = RecurrentRoBERTa(
            num_classes=config['num_classes'],
            window_size=config['window_size'],
            num_repeats=config['num_repeats'],
            max_windows=config['max_windows'],
            dropout=config['dropout']
        )
        
        # 5. 加载模型权重
        model.load_state_dict(ckpt['model_state_dict'])
        if device:
            model = model.to(device)
            
        # 6. 组装训练状态
        training_state = {
            'optimizer_state_dict': ckpt.get('optimizer_state_dict'),
            'scheduler_state_dict': ckpt.get('scheduler_state_dict'),
            'epoch': ckpt.get('epoch'),
            'train_losses': ckpt.get('train_losses'),
            'val_losses': ckpt.get('val_losses'),
            'train_accuracies': ckpt.get('train_accuracies'),
            'val_accuracies': ckpt.get('val_accuracies'),
            'best_val_loss': metrics.get('best_val_loss'),
            'best_accuracy': metrics.get('best_accuracy'),
            'training_config': training_config
        }
        
        logger.info(f"模型加载完成，当前epoch: {training_state['epoch']}")
        # 强制转换为int类型
        label2id = {int(k): int(v) for k, v in label2id.items()} if label2id is not None else None
        id2label = {int(k): int(v) for k, v in id2label.items()} if id2label is not None else None
        return model, config, label2id, id2label, training_state
        
    except Exception as e:
        raise ModelError(f"模型及配置加载失败: {e}")


# prepare_test_loader 函数已移至data_utils.dataset模块


@error_handler(error_type=ResourceError)
def get_device(device_type=None):
    """
    获取计算设备，并验证其可用性
    
    Returns:
        torch.device: 计算设备(GPU或CPU)
        
    Raises:
        ResourceError: 如果设备初始化失败
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 优先使用 device_type 指定的设备
        if device_type is not None:
            try:
                device = torch.device(device_type)
                logger.info(f"优先使用指定设备: {device}")
                if device.type == 'cuda' and not torch.cuda.is_available():
                    logger.warning("指定了CUDA但未检测到可用GPU，回退到CPU")
                    device = torch.device('cpu')
            except Exception as e:
                logger.warning(f"指定设备类型无效({device_type})，回退自动检测: {str(e)}")
                device = None
        else:
            device = None

        # 自动检测
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"可用GPU数量: {device_count}, 设备名称: {device_name}")
                try:
                    x = torch.randn(10, 10).to(device)
                    logger.info(f"成功初始化CUDA设备: {device}")
                    log_gpu_usage("初始化后")
                except Exception as e:
                    logger.error(f"GPU测试失败: {str(e)}")
                    logger.warning("回退到CPU设备")
                    device = torch.device('cpu')
            else:
                device = torch.device('cpu')
                logger.info("未检测到CUDA设备，使用CPU")
        logger.info(f"最终使用设备: {device}")
        return device
    except Exception as e:
        logger.error(f"设备初始化错误: {str(e)}")
        logger.warning("强制使用CPU")
        return torch.device('cpu')
