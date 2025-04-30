"""
评估模块 - 实现模型评估和测试逻辑
"""

import os
import torch
import logging
import numpy as np
import traceback
import argparse
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 测试模式设置（用于快速测试代码的健壮性）
TEST_MODE = False  # 设置为True时只处理少量样本
TEST_SAMPLES = 10  # 快速测试时处理的样本数量

# 导入评估工具
from evaluation_utils import evaluate_model
# predict 已移除，无需导入

from config import config
from utils import (
    setup_logging, get_device, create_model, 
    error_handler, validate_params, clean_gpu_memory, log_gpu_usage,
    ModelError, DataError,
    load_model_and_config
)
from data_utils.dataset import prepare_test_loader
# evaluate_model 函数已移至 evaluation_utils.py


@error_handler(error_type=ModelError, reraise=False)
@validate_params()
def predict_single(model, tokenizer, text, device, max_length=512, label_encoder=None):
    """
    对单个文本样本进行预测
    
    Args:
        model (nn.Module): 模型实例
        tokenizer: 分词器
        text (str): 文本样本
        device (torch.device): 计算设备
        max_length (int): 最大序列长度
        label_encoder (LabelEncoder, optional): 标签编码器
        
    Returns:
        tuple: (预测标签, 预测概率字典)
    """
    logger = logging.getLogger(__name__)
    
    try:
        model.eval()
        
        # 使用RoBERTa分词器对输入文本进行处理
        encoding = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
        
        # 移动到计算设备
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 处理不同类型的输出格式
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
        
        # 获取预测结果
        probs = logits.cpu().numpy().squeeze()
        pred_class = int(np.argmax(probs))
        
        # 如果提供了标签编码器，则转换为类别名称
        if label_encoder is not None:
            try:
                pred_class = label_encoder.inverse_transform([pred_class])[0]
            except Exception as e:
                logger.warning(f"标签转换失败: {str(e)}")
        
        # 返回预测类别和概率
        return pred_class, {
            'class': pred_class,
            'probability': float(np.max(probs)),
            'all_probs': {i: float(p) for i, p in enumerate(probs)}
        }
        
    except Exception as e:
        logger.error(f"预测单个样本时出错: {str(e)}")
        # 返回默认值，防止整个程序崩溃
        return -1, {
            'class': -1,
            'probability': 0.0,
            'all_probs': {},
            'error': str(e)
        }


@error_handler(error_type=ModelError, reraise=False)
@validate_params()
def process_long_text(model, tokenizer, text, max_windows=16, window_size=512, device=None):
    """
    处理长文本，将其分成多个窗口并汇总结果
    
    Args:
        model (nn.Module): 模型实例
        tokenizer: 分词器
        text (str): 长文本
        max_windows (int): 最大窗口数
        window_size (int): 每个窗口的大小
        device (torch.device): 计算设备
        
    Returns:
        dict: 各个类别的预测概率
    """
    logger = logging.getLogger(__name__)
    
    # 如果没有指定计算设备，则获取可用设备
    if device is None:
        device = get_device()
    
    try:
        model.eval()
        
        # 记录初始GPU内存使用情况
        if device.type == 'cuda':
            log_gpu_usage("处理长文本开始前")
        
        # 将文本拆分成词元
        try:
            tokens = tokenizer.tokenize(text)
        except Exception as e:
            logger.error(f"分词失败: {str(e)}")
            raise DataError(f"文本分词失败: {str(e)}")
        
        # 如果文本长度足够短，直接处理整个文本
        if len(tokens) <= window_size:
            return predict_single(model, tokenizer, text, device, window_size)
        
        # 否则将文本分成多个窗口
        windows = []
        for i in range(0, len(tokens), window_size):
            if len(windows) >= max_windows:
                logger.info(f"达到最大窗口数限制({max_windows})，截断剩余文本")
                break
            window_tokens = tokens[i:i+window_size]
            window_text = tokenizer.convert_tokens_to_string(window_tokens)
            windows.append(window_text)
        
        logger.info(f"长文本已分割为 {len(windows)} 个窗口")
        
        # 处理每个窗口
        all_probs = []
        successful_windows = 0
        
        for i, window in enumerate(tqdm(windows, desc="Processing windows")):
            try:
                _, window_result = predict_single(model, tokenizer, window, device, window_size)
                
                # 检查结果是否有错误标记
                if 'error' in window_result:
                    logger.warning(f"窗口 {i+1} 处理出错: {window_result['error']}")
                    continue
                    
                all_probs.append(window_result['all_probs'])
                successful_windows += 1
            except Exception as e:
                logger.warning(f"窗口 {i+1} 处理失败: {str(e)}")
                continue
        
        # 如果没有成功处理任何窗口，则返回错误
        if successful_windows == 0:
            logger.error("所有窗口处理均失败")
            return -1, {
                'class': -1,
                'probability': 0.0,
                'all_probs': {},
                'error': "所有窗口处理均失败"
            }
        
        # 将所有窗口的预测概率合并
        combined_probs = {}
        for probs in all_probs:
            for cls, prob in probs.items():
                if cls not in combined_probs:
                    combined_probs[cls] = 0
                combined_probs[cls] += prob
        
        # 标准化并找到最可能的类
        total = sum(combined_probs.values())
        if total > 0:  # 避免除以0
            for cls in combined_probs:
                combined_probs[cls] /= total
        
        # 找出概率最高的类别
        if combined_probs:
            pred_class = max(combined_probs, key=combined_probs.get)
            result_prob = combined_probs[pred_class]
        else:
            logger.warning("没有有效的预测概率")
            pred_class = -1
            result_prob = 0.0
        
        # 清理GPU内存
        if device.type == 'cuda':
            log_gpu_usage("处理长文本结束后")
            clean_gpu_memory()
        
        return pred_class, {
            'class': pred_class,
            'probability': result_prob,
            'all_probs': combined_probs,
            'processed_windows': successful_windows,
            'total_windows': len(windows)
        }
        
    except Exception as e:
        logger.error(f"处理长文本时出错: {str(e)}")
        # 尝试清理GPU内存
        if device and hasattr(device, 'type') and device.type == 'cuda':
            clean_gpu_memory()
        return -1, {
            'class': -1,
            'probability': 0.0,
            'all_probs': {},
            'error': str(e)
        }


def create_evaluation_directory(base_dir=None, model_dir=None, timestamp_format="%Y%m%d_%H%M%S"):
    """
    创建评估结果保存目录
    
    Args:
        base_dir (str, optional): 基础目录路径
        model_dir (str, optional): 模型目录名称，用于命名评估目录
        timestamp_format (str, optional): 时间戳格式
        
    Returns:
        str: 创建的目录路径
    """
    logger = logging.getLogger(__name__)
    
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    
    if model_dir:
        # 从模型路径中提取目录名
        model_name = os.path.basename(os.path.dirname(model_dir))
        eval_dir_name = f"eval_{model_name}_{timestamp}"
    else:
        eval_dir_name = f"eval_{timestamp}"
    
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "models", "evaluations")
    
    # 确保基础目录存在
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        logger.debug(f"创建基础目录: {base_dir}")
        
    # 创建评估目录
    eval_dir = os.path.join(base_dir, eval_dir_name)
    os.makedirs(eval_dir, exist_ok=True)
    
    logger.info(f"创建评估结果目录: {eval_dir}")
    return eval_dir


def save_evaluation_results(results, output_dir):
    """
    保存评估结果到文件
    
    Args:
        results (dict): 评估结果字典
        output_dir (str): 输出目录
    """
    logger = logging.getLogger(__name__)
    
    # 保存结果为文本文件
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("ReBerta模型评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"准确率 (Accuracy): {results['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {results['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {results['recall']:.4f}\n")
        f.write(f"F1分数 (F1 Score): {results['f1']:.4f}\n")
        
        if 'per_class_metrics' in results:
            f.write("\n类别详细指标:\n")
            for class_name, metrics in results['per_class_metrics'].items():
                f.write(f"  类别 '{class_name}':\n")
                f.write(f"    精确率: {metrics['precision']:.4f}\n")
                f.write(f"    召回率: {metrics['recall']:.4f}\n")
                f.write(f"    F1分数: {metrics['f1']:.4f}\n")
    
    logger.info(f"评估结果已保存到: {results_file}")
    
    # 生成性能指标图表
    if 'per_class_metrics' in results:
        try:
            # 创建性能对比图
            plt.figure(figsize=(12, 8))
            
            # 准备数据
            class_names = list(results['per_class_metrics'].keys())
            precision_values = [results['per_class_metrics'][c]['precision'] for c in class_names]
            recall_values = [results['per_class_metrics'][c]['recall'] for c in class_names]
            f1_values = [results['per_class_metrics'][c]['f1'] for c in class_names]
            
            # 绘制柱状图
            x = np.arange(len(class_names))
            width = 0.25
            
            plt.bar(x - width, precision_values, width, label='精确率')
            plt.bar(x, recall_values, width, label='召回率')
            plt.bar(x + width, f1_values, width, label='F1分数')
            
            plt.xlabel('类别')
            plt.ylabel('分数')
            plt.title('各类别性能指标对比')
            plt.xticks(x, class_names, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            metrics_plot_path = os.path.join(output_dir, "class_metrics.png")
            plt.savefig(metrics_plot_path)
            plt.close()
            
            logger.info(f"性能指标图表已保存到: {metrics_plot_path}")
        except Exception as e:
            logger.error(f"生成性能指标图表时出错: {str(e)}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ReBerta模型评估工具')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='要评估的模型文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='评估结果输出目录，默认会创建一个新目录')
    parser.add_argument('--batch_size', type=int, default=config['batch_size'], 
                        help='批次大小')
    parser.add_argument('--window_size', type=int, default=config['window_size'], 
                        help='滑动窗口大小')
    parser.add_argument('--num_repeats', type=int, default=config['num_repeats'], 
                        help='循环次数')
    parser.add_argument('--max_windows', type=int, default=config['max_windows'], 
                        help='最大窗口数')
    parser.add_argument('--test_path', type=str, default='data/validation',
                        help='测试数据路径')
    parser.add_argument('--mixed_precision', type=lambda x: (str(x).lower() == 'true'), 
                      default=config.get('mixed_precision', False), 
                      help='是否启用混合精度')
    parser.add_argument('--mixed_precision_type', type=str, 
                      default=config.get('mixed_precision_type', 'fp16'),
                      choices=['fp16', 'bf16'],
                      help='混合精度类型')
    parser.add_argument('--dropout', type=float, default=config.get('dropout', 0.1), 
                      help='dropout比例')
    args = parser.parse_args()
    
    # 验证模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        exit(1)
    
    # 创建输出目录
    eval_dir = args.output_dir
    if eval_dir is None:
        eval_dir = create_evaluation_directory(model_dir=args.model_path)
    else:
        # 确保目录存在
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(eval_dir, "evaluation.log")
    logger = logging.getLogger(__name__)
    
    # 记录执行环境
    logger.info("启动评估模块")
    logger.info(f"评估模型: {args.model_path}")
    logger.info(f"输出目录: {eval_dir}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    
    # 准备测试数据
    logger.info(f"加载测试数据集: {args.test_path}")
    test_loader, num_classes = prepare_test_loader(
        args.batch_size,
        config['max_len'],
        config['model_name'],
        args.test_path,
        max_samples=TEST_SAMPLES if TEST_MODE else None
    )   
        
    try:
        # 获取计算设备
        device = get_device()

        # 加载模型和配置（包含label2id/id2label）
        logger.info(f"加载模型和配置: {args.model_path}")
        model, config_ckpt, label2id, id2label = load_model_and_config(args.model_path, device)
        logger.info("模型和配置加载成功")

        # 检查标签映射
        if label2id is None or id2label is None:
            logger.error("未能在模型权重中找到label2id/id2label映射，无法保证标签一致性！")
            raise ValueError("模型权重缺少label2id/id2label，请检查训练和保存流程。")
        label_names = [id2label[str(i)] if str(i) in id2label else id2label[i] for i in range(len(id2label))]
        logger.info(f"标签类别: {', '.join(label_names)}")

        # 加载测试数据（强制使用训练时的label2id）
        logger.info(f"加载测试数据集: {args.test_path}")
        test_loader, num_classes = prepare_test_loader(
            args.batch_size,
            config['max_len'],
            config['model_name'],
            args.test_path,
            label2id=label2id,
            max_samples=TEST_SAMPLES if TEST_MODE else None
        )

        # 评估模型
        logger.info("开始评估模型...")
        eval_results = evaluate_model(model, test_loader, device, label_names, eval_dir)

        # 保存评估结果
        if 'error' not in eval_results:
            save_evaluation_results(eval_results, eval_dir)
            logger.info(f"评估结果摘要:")
            logger.info(f"准确率: {eval_results['accuracy']:.4f}")
            logger.info(f"精确率: {eval_results['precision']:.4f}")
            logger.info(f"召回率: {eval_results['recall']:.4f}")
            logger.info(f"F1分数: {eval_results['f1']:.4f}")
            logger.info(f"详细结果已保存到: {eval_dir}")
        else:
            logger.error(f"评估过程中发生错误: {eval_results['error']}")

    except Exception as e:
        logger.error(f"评估过程中出错: {str(e)}")
        logger.error(traceback.format_exc())

    finally:
        # 清理资源
        logger.info("评估模块执行完毕，清理资源...")
        if 'device' in locals() and hasattr(device, 'type') and device.type == 'cuda':
            clean_gpu_memory()

