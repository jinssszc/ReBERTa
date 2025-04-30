"""
ReBerta模型评估脚本

使用项目的现有功能和结构，评估已修复的checkpoint模型权重
"""

import os
import sys
import argparse
import logging
import torch
import numpy
import numpy as np
from torch.serialization import safe_globals
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.append(os.path.abspath('.'))

# 导入项目模块
from config import config
from models.reberta import RecurrentRoBERTa
from evaluation_utils import evaluate_model
from data_utils.dataset import prepare_test_loader

# 导入错误处理工具
from utils import (setup_logging, get_device, error_handler, clean_gpu_memory, log_gpu_usage,
                  ReBertaError, DataError, ModelError)


def parse_args():
    parser = argparse.ArgumentParser(description='ReBerta模型评估')
    parser.add_argument('--model_path', type=str, 
                        default='models/saved/reberta_ver2_trained_0416/best_accuracy_model_with_label_map.pt',
                        help='模型权重文件路径')
    parser.add_argument('--test_path', type=str, 
                        default='data/evaluation',
                        help='测试数据路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--output_dir', type=str, default='results', help='评估结果输出目录')
    parser.add_argument('--max_samples', type=int, default=None, help='最大测试样本数，默认为全部')
    return parser.parse_args()


@error_handler(error_type=ReBertaError, reraise=False)
def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(args.output_dir, "evaluate.log")
    
    # 获取设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 加载模型checkpoint
    logger.info(f"加载模型权重: {args.model_path}")
    
    # 导入更多numpy类型
    import numpy as np
    from numpy.core import numeric
    from numpy import ndarray
    
    # 扩展安全类型列表
    safe_types = [
        numpy._core.multiarray.scalar, 
        numpy.dtype,
        np.float64,
        np.int64,
        np.float32,
        np.int32,
        np.bool_,
        np.dtypes.Float64DType,
        numeric.dtype,
        ndarray
    ]
    
    # 使用weights_only=False加载完整模型
    with safe_globals(safe_types):
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    
    # 检查checkpoint结构
    logger.info(f"Checkpoint内容关键字: {list(checkpoint.keys())}")
    
    # 检查是否有label2id映射
    if 'label2id' not in checkpoint or 'id2label' not in checkpoint:
        raise ValueError("模型权重中缺少label2id或id2label映射，请先运行修复脚本")
    
    # 提取必要信息
    config_from_ckpt = checkpoint['config']
    label2id = checkpoint['label2id']
    id2label = checkpoint['id2label']
    num_classes = len(label2id)
    
    # 应用训练参数
    train_params = {
        'window_size': 256,
        'num_repeats': 2,
        'max_windows': 16,
        'dropout': 0.15,
    }
    
    # 合并参数到配置中
    for key, value in train_params.items():
        if key in config_from_ckpt:
            logger.info(f"使用参数: {key}={value}")
            config_from_ckpt[key] = value
    
    # 创建模型
    logger.info(f"创建RecurrentRoBERTa模型，类别数: {num_classes}")
    model = RecurrentRoBERTa(
        num_classes=num_classes,
        window_size=config_from_ckpt['window_size'],
        max_windows=config_from_ckpt['max_windows'],
        num_repeats=config_from_ckpt['num_repeats'],
        pretrained_path=config_from_ckpt.get('model_path', None),
        dropout=config_from_ckpt['dropout']
    )
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        logger.info("从model_state_dict加载模型权重...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning("没有找到model_state_dict，检查其他可能的键...")
        # 尝试其他可能的键名
        if 'state_dict' in checkpoint:
            logger.info("从state_dict加载模型权重...")
            model.load_state_dict(checkpoint['state_dict'])
        else:
            possible_keys = [k for k in checkpoint.keys() if 'state' in k.lower()]
            if possible_keys:
                logger.info(f"尝试从{possible_keys[0]}加载模型权重...")
                model.load_state_dict(checkpoint[possible_keys[0]])
            else:
                raise ValueError("找不到可用的模型权重，请检查checkpoint文件")
    
    model.to(device)
    model.eval()
    
    # 从标签映射创建标签编码器
    logger.info("从模型的label2id映射创建LabelEncoder...")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(list(id2label.values()))
    
    # 验证标签编码器
    logger.info(f"标签编码器创建成功，包含{len(label_encoder.classes_)}个类别")
    for i, class_name in enumerate(label_encoder.classes_[:5]):
        logger.info(f"  样例类别{i}: {class_name}")
    
    # 加载测试数据
    logger.info(f"从{args.test_path}加载测试数据...")
    try:
        test_loader, test_num_classes = prepare_test_loader(
            batch_size=args.batch_size,
            max_len=config_from_ckpt['max_len'],
            model_name=config_from_ckpt['model_name'],
            test_path=args.test_path,
            label_encoder=label_encoder,  # 传递创建的标签编码器
            max_samples=args.max_samples
        )
        
        # 确认测试数据集类别数与模型类别数匹配
        if test_num_classes != num_classes:
            logger.warning(f"警告：测试数据集类别数({test_num_classes})与模型类别数({num_classes})不匹配")
        
        # 获取标签名称
        label_names = list(id2label.values())
        
        # 评估模型
        logger.info("开始评估模型...")
        eval_results = evaluate_model(
            model=model,
            data_loader=test_loader,  # 修正参数名为data_loader
            device=device,
            label_names=label_names,
            log_dir=args.output_dir   # 修正参数名为log_dir
        )
        
        # 打印评估结果
        logger.info(f"评估结果:")
        logger.info(f"  准确率: {eval_results['accuracy']:.4f}")
        logger.info(f"  精确率: {eval_results['precision']:.4f}")
        logger.info(f"  召回率: {eval_results['recall']:.4f}")
        logger.info(f"  F1分数: {eval_results['f1']:.4f}")
        
        # 清理GPU内存
        if device.type == 'cuda':
            clean_gpu_memory()
        
        return eval_results
        
    except Exception as e:
        logger.error(f"评估过程中发生错误: {str(e)}")
        if device.type == 'cuda':
            clean_gpu_memory()
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)
