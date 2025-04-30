"""
数据集模块 - 负责数据加载和预处理
"""

import os
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import RobertaTokenizer
from tqdm import tqdm
import logging
import traceback
from pathlib import Path

# 导入错误处理工具
from utils import (
    ReBertaError, DataError, ConfigError, ResourceError,
    error_handler, validate_params, clean_gpu_memory
)

# 定义数据特定错误子类
class DataFormatError(DataError):
    """数据格式不正确错误"""
    pass

class DataLoadError(DataError):
    """数据加载失败错误"""
    pass


class LongTextDataset(Dataset):
    """
    长文本数据集类
    
    该类用于加载和预处理长文本数据，使用RoBERTa分词器进行文本编码
    """
    _INPUT_FIELD = 'text'   # 样本输入字段名
    _LABEL_FIELD = 'label'  # 标签字段名
    @error_handler(error_type=DataError)
    @validate_params(max_len=int)
    def __init__(self, file_paths, max_len, model_name="roberta-base", label_encoder=None):
        """
        初始化数据集
        
        Args:
            file_paths (list): 数据集文件路径列表
            max_len (int): 最大序列长度
            model_name (str): 使用的RoBERTa模型名称
            
        Raises:
            DataError: 数据加载或处理失败
            ConfigError: 参数无效
        """
        self.logger = logging.getLogger(__name__)
        
        if not file_paths:
            raise ConfigError("文件路径列表不能为空")
        if max_len <= 0:
            raise ConfigError(f"最大序列长度必须为正数，当前值: {max_len}")
            
        try:
            self.data = self.load_data(file_paths)
            if self.data.empty:
                raise DataFormatError(f"加载的数据为空")
                
            self.max_len = max_len
            
            # 初始化标签编码器（支持外部传入）
            self.logger.info("初始化标签编码器...")
            if self._LABEL_FIELD not in self.data.columns:
                raise DataFormatError(f"数据中缺少'{self._LABEL_FIELD}'列")
            
            if label_encoder is None:
                # 训练集：创建并fit
                unique_labels = self.data[self._LABEL_FIELD].unique()
                if len(unique_labels) == 0:
                    raise DataFormatError("没有可用的标签")
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(unique_labels)
            else:
                # 验证/测试集：直接使用传入的label_encoder
                self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(self.data[self._LABEL_FIELD])
            
            # 存储原始文本和标签
            if self._INPUT_FIELD not in self.data.columns:
                raise DataFormatError(f"数据中缺少'{self._INPUT_FIELD}'列")
                
            self.abstracts = self.data[self._INPUT_FIELD].tolist()
            self.labels = torch.tensor(self.labels, dtype=torch.long)
            
            # 优先使用本地分词器，如果不存在则从Hugging Face下载
            try:
                tokenizer_path = 'models/pretrained' if os.path.exists('models/pretrained') else model_name
                self.logger.info(f"加载分词器: {tokenizer_path}")
                self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                self.logger.error(f"加载分词器失败: {str(e)}")
                raise DataError(f"无法加载分词器: {str(e)}")
                
            self.logger.info(f"数据集初始化完成，样本数: {len(self.data)}, 类别数: {len(self.label_encoder.classes_)}")
        except Exception as e:
            if not isinstance(e, (DataError, ConfigError)):
                raise DataError(f"数据集初始化失败: {str(e)}") from e
            raise

    @error_handler(error_type=DataLoadError)
    def load_data(self, file_paths):
        """
        加载数据集，支持HuggingFace磁盘格式和JSONL格式（arXiv等）
        
        Args:
            file_paths (list): 数据集文件路径列表
        Returns:
            pandas.DataFrame: 合并后的数据集
        Raises:
            DataLoadError: 数据加载失败
        """
        import json
        all_data = []
        for dir_path in file_paths:
            self.logger.info(f"正在加载数据集: {dir_path}")
            if not os.path.exists(dir_path):
                raise DataLoadError(f"数据集路径不存在: {dir_path}")
            try:
                if dir_path.endswith('.jsonl'):
                    # 处理JSONL格式
                    records = []
                    with open(dir_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip() == '':
                                continue
                            try:
                                obj = json.loads(line)
                                # 兼容arXiv格式：text->text, label->label
                                text = obj.get(self._INPUT_FIELD, None)
                                label = obj.get(self._LABEL_FIELD, None)
                                if text is not None and label is not None:
                                    records.append({self._INPUT_FIELD: text, self._LABEL_FIELD: label})
                            except Exception as e:
                                self.logger.warning(f"跳过无法解析的行: {str(e)}")
                    df = pd.DataFrame(records)
                    if df.empty:
                        self.logger.warning(f"数据集为空: {dir_path}")
                        continue
                    # 清洗数据：移除text为空的样本
                    before_clean = len(df)
                    df = df.dropna(subset=[self._INPUT_FIELD])
                    after_clean = len(df)
                    if before_clean > after_clean:
                        self.logger.warning(f"移除了{before_clean - after_clean}条{self._INPUT_FIELD}为空的样本")
                    if df.empty:
                        self.logger.warning(f"清洗后数据集为空: {dir_path}")
                        continue
                    all_data.append(df)
                else:
                    # 默认HuggingFace磁盘格式
                    from datasets import load_from_disk
                    dataset = load_from_disk(dir_path)
                    df = dataset.to_pandas()
                    if df.empty:
                        self.logger.warning(f"数据集为空: {dir_path}")
                        continue
                    before_clean = len(df)
                    df = df.dropna(subset=[self._INPUT_FIELD])
                    after_clean = len(df)
                    if before_clean > after_clean:
                        self.logger.warning(f"移除了{before_clean - after_clean}条{self._INPUT_FIELD}为空的样本")
                    if df.empty:
                        self.logger.warning(f"清洗后数据集为空: {dir_path}")
                        continue
                    all_data.append(df)
            except Exception as e:
                self.logger.error(f"加载数据集{dir_path}时出错: {str(e)}")
                raise DataLoadError(f"无法加载数据集{dir_path}: {str(e)}")
        if not all_data:
            raise DataLoadError("没有成功加载任何数据")
        return pd.concat(all_data, ignore_index=True)


    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    @error_handler(error_type=DataError)
    def __getitem__(self, idx):
        """
        获取数据集中的样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (input_ids, attention_mask, token_type_ids, label)
            
        Raises:
            DataError: 获取样本失败
            IndexError: 索引超出范围
        """
        # 检查索引有效性
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"索引{idx}超出范围[0, {len(self.data)-1}]")
            
        try:
            # 获取文本
            text = self.abstracts[idx]  # 字段名不变，内容为 _INPUT_FIELD
            
            # 使用RoBERTa分词器编码文本
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt'
            )
            
            # 移除批次维度
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            return {
                'input_ids': input_ids, 
                'attention_mask': attention_mask,
                'labels': self.labels[idx]
            }
        except Exception as e:
            if isinstance(e, IndexError):
                raise
            self.logger.error(f"获取样本{idx}失败: {str(e)}")
            raise DataError(f"无法获取样本{idx}: {str(e)}")


class LongTextBatchSampler:
    """
    长文本批次采样器
    
    按照文本长度对样本进行分桶，以减少填充对内存和计算的影响
    """
    @error_handler(error_type=ConfigError)
    @validate_params(batch_size=int)
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, max_samples=None):
        """
        初始化批次采样器
        
        Args:
            dataset (LongTextDataset): 数据集实例
            batch_size (int): 批次大小
            shuffle (bool): 是否打乱样本顺序
            drop_last (bool): 是否丢弃最后不完整的批次，但为确保CUDA兼容性，对于采样场景将始终丢弃不完整批次
            max_samples (int): 最大样本数量限制
            
        Raises:
            ConfigError: 配置无效
        """
        self.logger = logging.getLogger(__name__)
        
        if not hasattr(dataset, 'abstracts'):
            raise ConfigError("数据集必须是LongTextDataset类型")
        if batch_size <= 0:
            raise ConfigError(f"批次大小必须为正数，当前值: {batch_size}")
            
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 当使用采样时，始终丢弃不完整批次以确保CUDA兼容性
        self.drop_last = True if max_samples is not None else drop_last
        
        # 调整max_samples为batch_size的整数倍，确保批次大小标准化
        self.max_samples = max_samples
        if self.max_samples is not None:
            original_samples = self.max_samples
            # 向下取整到最近的batch_size倍数
            self.max_samples = (self.max_samples // self.batch_size) * self.batch_size
            if self.max_samples == 0 and original_samples > 0:  # 防止max_samples小于batch_size的情况
                self.max_samples = self.batch_size
                self.logger.info(f"采样数量 {original_samples} 小于批次大小，已调整为批次大小: {self.batch_size}")
            elif self.max_samples != original_samples:
                self.logger.info(f"已将采样数量从 {original_samples} 调整为批次大小的整数倍: {self.max_samples}")
        
        try:
            # 获取所有样本的索引
            all_indices = list(range(len(dataset.abstracts)))  # 字段名不变，实际内容为 _INPUT_FIELD
            
            # 如果需要采样，首先设置随机种子并进行随机采样
            if self.max_samples is not None and self.max_samples < len(all_indices):
                # 设置随机种子，保证可重复性
                seed = 42  # 使用默认种子
                import torch
                import numpy as np
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                self.logger.info(f"采样器随机种子设置为: {seed}")
                
                # 随机采样指定数量的样本索引
                import random
                random.seed(seed)
                random.shuffle(all_indices)
                selected_indices = all_indices[:self.max_samples]
                self.logger.info(f"从{len(all_indices)}个样本中随机采样{len(selected_indices)}个样本")
            else:
                # 使用全部样本
                selected_indices = all_indices
                self.logger.info(f"使用全部{len(selected_indices)}个样本")
            
            # 只计算已选择样本的长度
            self.lengths = {}  # 使用字典存储索引到长度的映射
            for idx in tqdm(selected_indices, desc="计算已选择样本的文本长度"):
                # 计算分词后的实际长度（不执行分词以加速处理）
                # 这里使用一个粗略的估计
                text = dataset.abstracts[idx]
                length = min(len(text.split()), dataset.max_len)
                self.lengths[idx] = length
            
            # 按长度排序已选择的样本索引
            self.sorted_indices = sorted(selected_indices, key=lambda i: self.lengths[i])
            
            # 创建批次
            self.batches = []
            
            for i in range(0, len(self.sorted_indices), self.batch_size):
                # 检查是否还有足够样本形成完整批次
                if i + self.batch_size > len(self.sorted_indices):
                    if not self.drop_last:  # 仅当不丢弃不完整批次时才添加
                        self.batches.append(self.sorted_indices[i:len(self.sorted_indices)])
                    continue
                
                # 添加一个完整批次
                self.batches.append(self.sorted_indices[i:i + self.batch_size])
                
            self.logger.info(f"批次采样器初始化完成，批次数: {len(self.batches)}")
        except Exception as e:
            self.logger.error(f"批次采样器初始化失败: {str(e)}")
            raise ConfigError(f"初始化批次采样器时出错: {str(e)}")
    
    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.batches)
        
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)


@error_handler(error_type=DataError)
def collate_fn(batch):
    """
    批次整理函数
    
    将一个批次的样本整理成模型所需的格式
    
    Args:
        batch (list): 一个批次的样本列表
        
    Returns:
        dict: 包含模型输入的字典
        
    Raises:
        DataError: 批次处理失败
    """
    logger = logging.getLogger(__name__)
    
    if not batch:
        raise DataError("批次不能为空")
        
    try:
        # 确保所有样本具有相同的键
        expected_keys = {'input_ids', 'attention_mask', 'labels'}
        for item in batch:
            if not all(key in item for key in expected_keys):
                missing_keys = expected_keys - set(item.keys())
                raise DataError(f"样本缺少必要的键: {missing_keys}")
        
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'labels': labels
        }
    except Exception as e:
        if not isinstance(e, DataError):
            logger.error(f"批次处理失败: {str(e)}")
            raise DataError(f"批次处理失败: {str(e)}")
        raise


@error_handler(error_type=DataError)
@validate_params(batch_size=int, max_len=int)
def prepare_data_loaders(batch_size, max_len, model_name, train_path, val_path, max_samples=None, train_max_samples=None, val_max_samples=None):
    """
    准备训练和验证数据加载器
    
    Args:
        batch_size (int): 批次大小
        max_len (int): 最大序列长度
        model_name (str): 模型名称
        train_path (str): 训练数据路径
        val_path (str): 验证数据路径
        max_samples (int, optional): 最大样本数量限制(兼容旧API)，作用于训练和验证集，默认为None（不限制）
        train_max_samples (int, optional): 训练集最大样本数量限制，优先级高于max_samples，默认为None（不限制）
        val_max_samples (int, optional): 验证集最大样本数量限制，优先级高于max_samples，默认为None（不限制）
        
    Returns:
        tuple: (train_loader, val_loader, num_classes) 训练和验证数据加载器及类别数
        
    Raises:
        DataError: 数据加载失败
        ConfigError: 参数无效
    """
    import logging
    import os
    from torch.utils.data import DataLoader
    
    logger = logging.getLogger(__name__)
    
    if batch_size <= 0:
        raise ConfigError(f"批次大小必须为正数，当前值: {batch_size}")
    if max_len <= 0:
        raise ConfigError(f"最大序列长度必须为正数，当前值: {max_len}")
    if not train_path:
        raise ConfigError("训练数据路径不能为空")
    if not val_path:
        raise ConfigError("验证数据路径不能为空")
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_full_path = os.path.join(base_dir, train_path)
    val_full_path = os.path.join(base_dir, val_path)
    
    # 检查路径是否存在
    if not os.path.exists(train_full_path):
        raise DataLoadError(f"训练数据路径不存在: {train_full_path}")
    if not os.path.exists(val_full_path):
        raise DataLoadError(f"验证数据路径不存在: {val_full_path}")
    
    # 处理采样参数优先级：专用参数 > 通用参数 > None
    # 对于训练集
    train_samples = train_max_samples if train_max_samples is not None else max_samples
    # 对于验证集
    val_samples = val_max_samples if val_max_samples is not None else max_samples
    
    # 记录采样设置
    logger.info(f"训练集采样设置: {train_samples if train_samples else '使用全部数据'}")
    logger.info(f"验证集采样设置: {val_samples if val_samples else '使用全部数据'}")
    
    try:
        # 创建训练数据集
        logger.info("加载训练数据集...")
        train_dataset = LongTextDataset(
            file_paths=[train_full_path],
            max_len=max_len,
            model_name=model_name
        )
        num_classes = len(train_dataset.label_encoder.classes_)
        label_encoder = train_dataset.label_encoder
        
        # 加载验证集，使用训练集的label_encoder
        val_dataset = LongTextDataset(
            file_paths=[val_full_path],
            max_len=max_len,
            model_name=model_name,
            label_encoder=label_encoder
        )
        
        # 创建批次采样器
        logger.info("创建批次采样器...")
        train_batch_sampler = LongTextBatchSampler(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            max_samples=train_samples
        )
        
        val_batch_sampler = LongTextBatchSampler(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            max_samples=val_samples
        )
        
        # 计算可用的最佳工作线程数
        import multiprocessing
        # 使用CPU逻辑核心数的一半，确保至少为1，Windows平台上默认为0可能导致性能问题
        optimal_workers = max(1, multiprocessing.cpu_count() // 2)
        logger.info(f"使用工作线程数: {optimal_workers}")
        
        # 创建数据加载器
        logger.info("创建数据加载器...")
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            num_workers=optimal_workers,  # 使用计算的最佳工作线程数
            pin_memory=True  # 启用固定内存，加速CPU到GPU的数据传输
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            collate_fn=collate_fn,
            num_workers=optimal_workers,  # 使用计算的最佳工作线程数
            pin_memory=True  # 启用固定内存，加速CPU到GPU的数据传输
        )
        
        logger.info(f"数据加载器准备完成，类别数: {num_classes}")
        return label_encoder, train_loader, val_loader, num_classes
    
    except Exception as e:
        if not isinstance(e, (DataError, ConfigError)):
            logger.error(f"准备数据加载器失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise DataError(f"准备数据加载器失败: {str(e)}")
        raise


@error_handler(error_type=DataError)
@validate_params(batch_size=int, max_len=int)
def prepare_test_loader(batch_size, max_len, model_name, test_path, label_encoder, max_samples=None):
    """
    准备测试数据加载器
    
    Args:
        batch_size (int): 批次大小
        max_len (int): 最大序列长度
        model_name (str): 模型名称
        test_path (str): 测试数据路径
        label_encoder: 训练集的标签编码器
        max_samples (int): 最大样本数量限制，用于快速测试模式，默认为None（不限制）
    
    Returns:
        tuple: (test_loader, num_classes) 测试数据加载器及类别数
        
    Raises:
        DataError: 数据加载失败
        ConfigError: 参数无效
    """
    import logging
    import os
    from torch.utils.data import DataLoader
    
    logger = logging.getLogger(__name__)
    
    if batch_size <= 0:
        raise ConfigError(f"批次大小必须为正数，当前值: {batch_size}")
    if max_len <= 0:
        raise ConfigError(f"最大序列长度必须为正数，当前值: {max_len}")
    if not test_path:
        raise ConfigError("测试数据路径不能为空")
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_full_path = os.path.join(base_dir, test_path)
    
    # 检查路径是否存在
    if not os.path.exists(test_full_path):
        raise DataLoadError(f"测试数据路径不存在: {test_full_path}")
    
    try:
        # 加载测试集，必须传入训练集的label_encoder
        test_dataset = LongTextDataset(
            file_paths=[test_full_path],
            max_len=max_len,
            model_name=model_name,
            label_encoder=label_encoder
        )
        num_classes = len(label_encoder.classes_)
        
        # 创建批次采样器
        logger.info("创建批次采样器...")
        test_batch_sampler = LongTextBatchSampler(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            max_samples=max_samples
        )
        
        # 计算可用的最佳工作线程数
        import multiprocessing
        # 使用CPU逻辑核心数的一半，确保至少为1，Windows平台上默认为0可能导致性能问题
        optimal_workers = max(1, multiprocessing.cpu_count() // 2)
        logger.info(f"使用工作线程数: {optimal_workers}")
        
        # 创建数据加载器
        logger.info("创建数据加载器...")
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_batch_sampler,
            collate_fn=collate_fn,
            num_workers=optimal_workers,  # 使用计算的最佳工作线程数
            pin_memory=True  # 启用固定内存，加速CPU到GPU的数据传输
        )
        
        logger.info(f"测试数据加载器准备完成，类别数: {num_classes}")
        return test_loader, num_classes
        
    except Exception as e:
        if not isinstance(e, (DataError, ConfigError)):
            logger.error(f"准备测试数据加载器失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise DataError(f"准备测试数据加载器失败: {str(e)}")
        raise
