"""
模型架构模块 - 定义基于RoBERTa的循环长文本分类模型

该模型使用标准RoBERTa编码器作为底层，通过滑动窗口方法处理长文本
"""

import os
import gc
import time
import torch
import traceback
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig
from torch.utils.checkpoint import checkpoint


class RecurrentRoBERTa(nn.Module):
    """
    循环RoBERTa模型
    
    该模型通过滑动窗口结合标准RoBERTa处理长文本，
    使用双层编码器架构和循环迭代处理来提升长文档的上下文理解能力。
    第一层编码器处理文本窗口，第二层编码器融合窗口级表示，实现全局理解。
    
    本文件用于处理一个批次的文本数据，并在运行中的批次循环中调用
    """
    
    def __init__(self, num_classes, window_size=514, num_repeats=2, max_windows=16, 
                 pretrained_path=None, dropout=0.1, label2id=None, id2label=None):
        """
        初始化模型
        
        Args:
            num_classes (int): 分类类别数
            window_size (int): 滑动窗口大小（token数）
            num_repeats (int): 迭代次数
            max_windows (int): 最大窗口数
            pretrained_path (str, optional): 预训练模型路径，如果提供则使用绝对路径加载
            dropout (float, optional): Dropout比例，用于模型正则化，默认为0.1
        """
        super().__init__()
        
        # 保存基本参数
        self.num_classes = num_classes
        if label2id is not None:
            # Fix: support both str and int keys in label2id
            # If keys are all int or digit-string, convert; else, keep as is
            if all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in label2id.keys()):
                self.label2id = {int(k): int(v) for k, v in label2id.items()}
            else:
                self.label2id = dict(label2id)
        else:
            self.label2id = {i: i for i in range(num_classes)}
        if id2label is not None:
            # Fix: support both int and str keys/values in id2label
            # If keys are all int or digit-string, convert key; else, keep as is
            if all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in id2label.keys()):
                self.id2label = {int(k): v for k, v in id2label.items()}
            else:
                self.id2label = dict(id2label)
        else:
            self.id2label = {i: i for i in range(num_classes)}

        # 一定要同步到 config（注意：此时self.roberta_config还未初始化，需后移到初始化之后）

        self.window_size = window_size
        self.num_repeats = num_repeats
        self.max_windows = max_windows
        
        try:
            # 确认是否有预训练路径
            if pretrained_path and os.path.exists(pretrained_path):
                # 使用提供的本地预训练模型路径
                model_path = pretrained_path
                print(f"使用本地预训练模型路径: {model_path}")
                
                # 加载配置和模型
                self.roberta_config = RobertaConfig.from_pretrained(model_path)
                self.first_encoder = RobertaModel.from_pretrained(
                    model_path, 
                    config=self.roberta_config,
                    local_files_only=True  # 始终使用本地文件
                )
            else:
                # 如果没有提供有效的预训练路径，使用基本配置初始化
                print("没有提供有效的预训练路径，使用基本配置初始化模型")
                self.roberta_config = RobertaConfig(vocab_size=50265)
                self.first_encoder = RobertaModel(self.roberta_config)
                model_path = None

            self.roberta_config.label2id = self.label2id
            self.roberta_config.id2label = self.id2label
            self.roberta_config.num_labels = self.num_classes
            
            # 添加用于窗口间[CLS]融合的门控机制
            self.window_gate = nn.Linear(self.roberta_config.hidden_size * 2, 1)

            # 同步label2id和id2label到config，保证保存权重时写入config.json
            self.roberta_config.label2id = self.label2id
            self.roberta_config.id2label = self.id2label
            
            # 添加用于全局表示注入的门控机制
            self.global_gate = nn.Linear(self.roberta_config.hidden_size * 2, 1)
            
            # 添加用于表示缩放和规范化的层
            self.layernorm1 = nn.LayerNorm(self.roberta_config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            
            # 第二层：RoBERTa编码器（仅使用其隐藏层和位置编码）
            # 从相同路径初始化第二个编码器
            try:
                # 确保也使用和第一编码器相同的模型路径
                if model_path:
                    second_encoder_config = RobertaConfig.from_pretrained(model_path)
                    self.second_encoder = RobertaModel.from_pretrained(
                        model_path, 
                        config=second_encoder_config,
                        local_files_only=True  # 始终使用本地文件
                    )
                else:
                    # 如果第一个编码器使用了基本配置，第二个也使用
                    second_encoder_config = RobertaConfig(vocab_size=50265)
                    self.second_encoder = RobertaModel(second_encoder_config)
                    
                print("第二个编码器加载成功")
            except Exception as e:
                print(f"加载第二个编码器失败: {str(e)}")
                # 如果第二个编码器加载失败，使用与第一个相同的配置
                second_encoder_config = self.roberta_config
                self.second_encoder = RobertaModel(second_encoder_config)
                print("使用基本配置初始化第二个编码器")
                
        except Exception as e:
            print(f"模型初始化失败: {str(e)}")
            # 使用基本配置强制初始化
            print("使用基本配置初始化模型")
            self.roberta_config = RobertaConfig(vocab_size=50265)
            self.first_encoder = RobertaModel(self.roberta_config)
            self.second_encoder = RobertaModel(self.roberta_config)
            
            # 初始化其他必要的组件
            self.window_gate = nn.Linear(self.roberta_config.hidden_size * 2, 1)
            self.global_gate = nn.Linear(self.roberta_config.hidden_size * 2, 1)
            self.layernorm1 = nn.LayerNorm(self.roberta_config.hidden_size)
            self.dropout = nn.Dropout(dropout)
        
        # 限制最大窗口数
        max_allowed_windows = 32  # 最大允许窗口数
        
        if max_windows > max_allowed_windows:
            # 请求的max_windows超过限制，截断到允许的最大值
            self.max_windows = max_allowed_windows  # 重置最大窗口数
        else:
            # 使用配置的max_windows值
            self.max_windows = max_windows
        
        # 第二层编码器的最大序列长度设置
        # 这设置足够处理最多32个窗口加上开头和结尾的特殊标记
        # 第二层编码器的作用是聚合来自第一层编码器的窗口表示
        self.second_encoder.config.max_position_embeddings = 64
        
        # 分类头
        # 添加pooler层，与BERT/RoBERTa标准实现一致
        self.pooler = nn.Sequential(
            nn.Linear(self.roberta_config.hidden_size, self.roberta_config.hidden_size),
            nn.Tanh()
        )
        
        # 初始化pooler权重，使用与BERT相同的初始化方式
        nn.init.normal_(self.pooler[0].weight, std=0.02)
        nn.init.zeros_(self.pooler[0].bias)
        
        # 初始化门控层权重，保持训练稳定性
        nn.init.normal_(self.window_gate.weight, std=0.02)
        nn.init.zeros_(self.window_gate.bias)
        nn.init.normal_(self.global_gate.weight, std=0.02)
        nn.init.zeros_(self.global_gate.bias)
        
        self.classifier = nn.Linear(self.roberta_config.hidden_size, num_classes)
        
        # 参数配置
        self.window_size = window_size
        self.num_repeats = num_repeats
        self.max_windows = max_windows
        
        # 是否支持FP32门控计算
        self.supports_autocast = hasattr(torch, 'autocast')
        
        # 在初始化RoBERTa编码器后添加以下代码来确保参数可训练
        # 确保第一编码器参数可训练
        for param in self.first_encoder.parameters():
            param.requires_grad = True
            
        # 确保第二编码器参数可训练
        for param in self.second_encoder.parameters():
            param.requires_grad = True
            
        # 确保分类头参数可训练
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            token_type_ids: token类型标识
            labels: 标签
        
        Returns:
            SequenceClassifierOutput: 包含loss和logits的标准transformers输出
        """
        try:
            batch_size, seq_len = input_ids.shape
            print(f"Input shape: batch_size={batch_size}, seq_len={seq_len}")
            
            # 初始化CLS表示和前一个窗口的CLS表示
            device = input_ids.device
            # 确保requires_grad=True
            cls_representation = torch.zeros(batch_size, self.roberta_config.hidden_size, device=device, requires_grad=True)
            prev_cls_representation = torch.zeros(batch_size, self.roberta_config.hidden_size, device=device, requires_grad=True)
            
            # 获取特殊标记的ID
            cls_token_id = int(self.first_encoder.config.bos_token_id)  # RoBERTa中<s>作为[CLS]
            sep_token_id = int(self.first_encoder.config.eos_token_id)  # RoBERTa中</s>作为[SEP]
            print(f"Special token IDs: CLS={cls_token_id}, SEP={sep_token_id}")
            
            # 使用最佳窗口大小 - 预留空间给特殊标记
            # 新窗口大小 = 原始大小 - 2（为[CLS]和[SEP]预留空间）
            content_window_size = min(self.window_size - 2, 512)  # 正文内容窗口大小
            total_window_size = content_window_size + 2  # 加上[CLS]和[SEP]
            
            # 使用attention_mask计算实际序列长度
            if attention_mask is not None:
                actual_seq_len = attention_mask.sum(dim=1).max().item()  # 批次中最长的实际长度
            else:
                actual_seq_len = seq_len
            
            # 使用实际长度估计窗口,考虑是否要启用微批处理
            estimated_windows = min(self.max_windows, 
                                     actual_seq_len // content_window_size + 
                                     (1 if actual_seq_len % content_window_size > 0 else 0))
            
            effective_num_repeats = self.num_repeats  # 使用配置的迭代次数
            print(f"Using content_window_size={content_window_size}, total_window_size={total_window_size}, repeats={effective_num_repeats}")
            
            # 定义标志变量 - 初始设置
            # 仅用于标记文本长度估计，不直接触发优化
            has_long_text = estimated_windows >= 8
            has_extra_long_text = estimated_windows >= 14
            # 初始化优化策略状态变量
            extreme_optimization = False  # 默认不启用预清理
            
            ### Recurrent Block
            for iteration in range(effective_num_repeats):
                print(f"Starting iteration {iteration+1}/{effective_num_repeats}")
                # 创建用于聚义层输入序列的存储所有窗口的[CLS]表示
                window_cls_list = []
                if iteration > 0:
                    prev_cls_representation = torch.zeros(batch_size, self.roberta_config.hidden_size, device=device)
                
                # 记录每个样本的有效窗口数量
                valid_window_counts = torch.ones(batch_size, dtype=torch.long, device=device) * self.max_windows
                
                # 梯度检查点模式标志
                # 根据预估窗口数量设置梯度检查点模式
                # 这里允许在第一次迭代就使用校验点以保证内存效率
                use_checkpointing = has_long_text
                
                # 设置优化标志，仅对非第一次迭代生效
                if iteration > 0 and 'max_valid_windows' in locals():
                    # 第二次及后续迭代根据上次处理的实际窗口数量决定
                    extreme_optimization = max_valid_windows >= 8
                else:
                    # 第一次迭代绝不启用预清理
                    extreme_optimization = False
                    
                # 设置微批处理参数
                micro_batch_mode = (extreme_optimization and iteration > 0) or has_extra_long_text
                if micro_batch_mode and (extreme_optimization and iteration > 0):
                    print("micro batch mode enabled for extreme_optimization")
                elif micro_batch_mode and has_extra_long_text:
                    print("micro batch mode enabled for has_extra_long_text")
                micro_batch_size = 2  # 每次处理2个窗口
                
                ### Layer 1 窗口层
                for window_idx, i in enumerate(range(0, seq_len, content_window_size)):
                    # 如果窗口数超过限制，停止处理
                    if window_idx >= self.max_windows:  # 限制窗口数量
                        print(f"Reached maximum window limit ({self.max_windows}), stopping")
                        break
                    
                    # 微批处理模式下的内存管理
                    if micro_batch_mode and window_idx > 0 and window_idx % micro_batch_size == 0:
                        # 强制同步所有正在运行的CUDA操作
                        torch.cuda.synchronize()
                        
                        # 释放临时变量
                        if 'token_embeddings' in locals():
                            del token_embeddings
                        if 'token_embeddings_copy' in locals():
                            del token_embeddings_copy
                        if 'outputs' in locals():
                            del outputs
                            
                        # 强制垃圾回收
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 监控内存使用情况
                        if window_idx % (micro_batch_size * 2) == 0:  # 仅每两个微批打印一次
                            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                            print(f"GPU memory after micro-batch {window_idx // micro_batch_size}: current={allocated:.2f}GB, peak={max_allocated:.2f}GB")
                
                    ### 制作窗口输入序列            
                    #从整个样本的全部序列中提取当前窗口的正文内容
                    end_idx = min(i + content_window_size, seq_len)
                    window_content = input_ids[:, i:end_idx] #冒号代表选择全部批次
                    
                    ### 检查窗口内padding比例判定当前窗口是否到达正文边界 - 仅从第二个窗口开始应用padding检测 - 如果只有第一个窗口则退化成原版Roberta
                    if attention_mask is not None and window_idx > 0:  
                        window_content_mask = attention_mask[:, i:end_idx]
                        
                        # 计算每个样本在当前窗口中的padding比例
                        padding_ratio = 1.0 - window_content_mask.float().mean(dim=1).clone()  # [batch_size]
                        
                        # 检查每个样本的padding比例是否超过阈值(60%)
                        padding_threshold = 0.6
                        excessive_padding = (padding_ratio > padding_threshold)
                        
                        # 如果有样本的padding比例超过阈值，记录它们的有效窗口数量
                        if excessive_padding.any():
                            # 对于超过阈值的样本，将这个窗口标记为它们的最后一个有效窗口
                            for batch_idx in torch.nonzero(excessive_padding).squeeze(-1):  # 确保正确处理单个元素
                                batch_idx_item = batch_idx.item()
                                valid_window_counts[batch_idx_item] = window_idx
                            
                            # 如果全部样本都超过padding阈值，可以直接退出循环
                            if excessive_padding.all():
                                print(f"All samples exceed padding threshold at window {window_idx}, stopping further window processing")
                                break
                
                    # 添加[CLS]和[SEP]标记
                    # 确保使用标量值而非张量
                    cls_tokens = torch.full((batch_size, 1), int(cls_token_id), device=device, dtype=input_ids.dtype)
                    sep_tokens = torch.full((batch_size, 1), int(sep_token_id), device=device, dtype=input_ids.dtype)
                    
                    # 组合成完整的窗口输入
                    window_input_ids = torch.cat([cls_tokens, window_content.clone(), sep_tokens], dim=1)
                    
                    # 为新的输入创建对应的attention_mask
                    if attention_mask is not None:
                        window_content_mask = attention_mask[:, i:end_idx].clone()  # 克隆避免原地修改
                        # [CLS]和[SEP]的mask设为1（注意）
                        special_mask = torch.ones((batch_size, 2), device=device, dtype=attention_mask.dtype)
                        window_attention_mask = torch.cat([special_mask[:, :1], window_content_mask, special_mask[:, 1:]], dim=1)
                    else:
                        # 如果没有mask，创建全为1的mask
                        window_attention_mask = torch.ones_like(window_input_ids)
                    
                    print(f"Processing window {window_idx+1}: content tokens {i}-{end_idx} (size: {end_idx-i}), "
                          f"with special tokens: {window_input_ids.shape[1]}")
                    
                    ### 实施门控机制，融合前一个窗口表示/前一个迭代的全局表示
                    # 处理输入序列的处理错误
                    try:
                        # 获取token嵌入
                        token_embeddings = self.first_encoder.embeddings.word_embeddings(window_input_ids)
                        
                        # 获取[CLS]位置的嵌入
                        cls_embedding = token_embeddings[:, 0].clone()
                        
                        # 如果不是第一个窗口，在处理前将前一个窗口的[CLS]表示融入到当前窗口
                        if window_idx > 0:
                            # 将上一个窗口的CLS表示进行缩放
                            scaled_prev_cls = self.dropout(self.layernorm1(prev_cls_representation))
                            
                            # 使用门控机制结合两个表示
                            concat_tensor = torch.cat([cls_embedding, scaled_prev_cls], dim=1)
                            gate = torch.sigmoid(self.window_gate(concat_tensor))
                            fused_cls_embedding = gate * cls_embedding + (1 - gate) * scaled_prev_cls
                            
                            # 保存门控融合后的CLS状态，用于后续残差连接
                            pre_forward_cls = fused_cls_embedding.clone()
                            
                            # 将融合后的[CLS]表示放回嵌入序列
                            token_embeddings_copy = token_embeddings.clone()
                            token_embeddings_copy[:, 0] = fused_cls_embedding
                        else:
                            # 第一个窗口的处理
                            # 如果是第二次及以后的迭代，引入全局表示
                            if iteration > 0:
                                try:
                                    # 去掉FP32转换，直接使用当前精度
                                    concat_global = torch.cat([cls_embedding, cls_representation], dim=1)
                                    global_gate = torch.sigmoid(self.global_gate(concat_global))
                                    fused_cls_embedding = global_gate * cls_embedding.clone() + (1 - global_gate) * cls_representation.clone()
                                    
                                    print(f"Combined with global representation, gate value: {global_gate.mean().item():.4f}")
                                    
                                    # 保存门控融合后的CLS状态，用于后续残差连接
                                    pre_forward_cls = fused_cls_embedding.clone()
                                    
                                    # 将融合后的[CLS]表示放回嵌入序列
                                    token_embeddings_copy = token_embeddings.clone()
                                    token_embeddings_copy[:, 0] = fused_cls_embedding
                                except Exception as e:
                                    print(f"Error combining with global representation: {str(e)}")
                                    print(f"Debug info - cls_embedding shape: {cls_embedding.shape}, dtype: {cls_embedding.dtype}")
                                    print(f"Debug info - cls_representation shape: {cls_representation.shape}, dtype: {cls_representation.dtype}")
                                    traceback.print_exc()
                                    # 发生错误时使用原始嵌入
                                    fused_cls_embedding = cls_embedding.clone()
                                    pre_forward_cls = cls_embedding.clone()
                                    token_embeddings_copy = token_embeddings.clone()
                            else:
                                # 第一次迭代不引入全局表示
                                fused_cls_embedding = cls_embedding.clone()
                                pre_forward_cls = cls_embedding.clone()
                                token_embeddings_copy = token_embeddings.clone()
                        
                        ### 窗口正式运算
                        # 前向传播
                        try:
                            # 如果使用校验点模式，则使用checkpoint包装编码器
                            if use_checkpointing:
                                # 导入checkpoint函数
                                from torch.utils.checkpoint import checkpoint
                                
                                # 定义包装了编码器的前向传播函数
                                def forward_func(inputs_embeds, attention_mask, token_type_ids):
                                    return self.first_encoder(
                                        inputs_embeds=inputs_embeds,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        output_hidden_states=True
                                    )
                                
                                # 使用梯度检查点包装函数
                                outputs = checkpoint(
                                    forward_func,
                                    token_embeddings_copy,
                                    window_attention_mask,
                                    token_type_ids,
                                    use_reentrant=False
                                )
                            else:
                                # 常规前向传播，使用当前精度
                                outputs = self.first_encoder(
                                    inputs_embeds=token_embeddings_copy,
                                    attention_mask=window_attention_mask,
                                    token_type_ids=token_type_ids,
                                    output_hidden_states=True
                                )
                            
                            # 成功处理后的操作
                            # 提取当前窗口的[CLS]表示 - 现在这是真正的[CLS]标记
                            cls_output = outputs.last_hidden_state[:, 0].clone()  # 克隆避免原地修改
                            
                            # 简化版残差连接 - 将前向传播前后的CLS表示结合
                            alpha = 0.8  # 残差缩放因子，可调整
                            current_cls = alpha * cls_output + (1 - alpha) * pre_forward_cls
                            #print(f"Applied residual connection with alpha={alpha}")
                            
                            # 保存当前窗口的[CLS]表示，作为下一个窗口的上下文
                            # 这里保存的是每个样本各自的CLS表示，确保样本间信息不混合
                            prev_cls_representation = current_cls
                            
                            # 更新当前窗口的[CLS]表示
                            if window_idx == 0:
                                cls_representation = current_cls
                            
                            # 将当前窗口的[CLS]表示添加到列表中
                            window_cls_list.append(current_cls)
                                
                        except Exception as e:
                            # 前向传播错误处理
                            import traceback
                            error_trace = traceback.format_exc()
                            error_msg = str(e)
                            print(f"Error in window forward pass: {error_msg}")
                            print(f"Window {window_idx} details: start={i}, end={end_idx}, size={end_idx-i}")
                            print(f"Input shape: {window_input_ids.shape}, attention mask shape: {window_attention_mask.shape}")
                            print(f"Error trace: {error_trace}")
                            
                            # 检测是否是显存不足错误
                            is_oom_error = "CUDA out of memory" in error_msg or "OOM" in error_msg
                            
                            # 如果是内存不足错误，执行特殊处理
                            if is_oom_error:
                                print(f"CUDA OOM detected in forward pass of window {window_idx}")
                                # 释放临时变量
                                locals_to_delete = ['token_embeddings', 'token_embeddings_copy', 'outputs', 'fused_cls_embedding']
                                for var_name in locals_to_delete:
                                    if var_name in locals():
                                        del locals()[var_name]
                                
                                # 清理资源
                                import gc
                                for _ in range(3):
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                
                                # 显示内存状态
                                if torch.cuda.is_available():
                                    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                                    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                                    print(f"GPU memory after OOM: current={allocated:.2f}GB, peak={max_allocated:.2f}GB")
                                
                                # 对于严重的OOM，特别是在微批处理模式下，考虑中断窗口处理
                                if micro_batch_mode and window_idx > 0:
                                    print(f"Severe OOM in micro-batch mode with {window_idx} windows processed. Stopping batch processing...")
                                    break  # 跳出window_idx循环
                            
                            # 使用备用方案，以避免计算图断裂
                            print(f"Using fallback representation for window {window_idx}")
                            # 使用更简单的方式处理嵌入
                            fallback_cls = self.dropout(self.layernorm1(cls_embedding.clone()))
                            fallback_cls.requires_grad_(True)  # 确保保持梯度流
                            
                            current_cls = fallback_cls
                            prev_cls_representation = current_cls
                            window_cls_list.append(current_cls)
                            continue
                    
                    except Exception as e:
                        # 嵌入处理错误，这是更基础的错误
                        import traceback
                        error_trace = traceback.format_exc()
                        error_msg = str(e)
                        print(f"Error in embedding processing: {error_msg}")
                        print(f"Window {window_idx} details: start={i}, end={end_idx}, size={end_idx-i}")
                        print(f"Error trace: {error_trace}")
                        
                        # 检测是否是显存不足错误
                        is_oom_error = "CUDA out of memory" in error_msg or "OOM" in error_msg
                        
                        # 调整所有样本的有效窗口计数，因为嵌入处理错误更严重
                        for batch_idx in range(batch_size):
                            valid_window_counts[batch_idx] = min(valid_window_counts[batch_idx].item(), window_idx)
                        
                        # 对于嵌入处理错误，可能需要更彻底的重置
                        if is_oom_error:
                            print(f"Critical CUDA OOM detected in embedding processing of window {window_idx}")
                            # 尝试彻底清理资源
                            import gc
                            for _ in range(5):  # 更多轮次清理
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            
                            # 显示内存状态
                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                                print(f"GPU memory after critical OOM: current={allocated:.2f}GB, peak={max_allocated:.2f}GB")
                            
                            # 在微批处理模式下，严重内存问题可能需要跳过整个批次
                            if micro_batch_mode:
                                print(f"Critical OOM in embedding processing, stopping batch processing")
                                break  # 跳出window_idx循环
                        
                        # 尝试最基本的备用方案 - 使用零向量
                        try:
                            print(f"Creating zero vector as fallback for window {window_idx}")
                            hidden_dim = self.first_encoder.config.hidden_size
                            zero_cls = torch.zeros(batch_size, hidden_dim, device=device, requires_grad=True)  # 确保需要梯度
                            
                            current_cls = zero_cls
                            prev_cls_representation = current_cls
                            window_cls_list.append(current_cls)
                        except Exception:
                            # 如果连备用方案都失败，可能需要中断整个批次处理
                            print("Fatal error, even zero vector creation failed, cannot process batch")
                            break  # 跳出window_idx循环
                        continue
                
                print(f"Processed {len(window_cls_list)} windows in iteration {iteration+1}")
                
                # 窗口层运算结束，清理内存
                if torch.cuda.is_available():
                    # 判断是否需要更彻底的清理
                    if micro_batch_mode and len(window_cls_list) >= 4:
                        print(f"Preparing memory for second encoder with {len(window_cls_list)} windows...")
                        import gc
                        gc.collect()  # 先回收Python对象
                        torch.cuda.empty_cache()  # 再清理GPU缓存
                        
                        # 显示内存状态
                        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        print(f"GPU memory before second encoder: current={allocated:.2f}GB, peak={max_allocated:.2f}GB")
                    else:
                        # 简单清理对于小批次或窗口数少的情况已足够
                        torch.cuda.empty_cache()
                
                # 如果没有处理窗口，跳过第二层编码器
                if len(window_cls_list) == 0:
                    print("No windows were processed successfully, skipping second encoder")
                    continue
                
                # 获取当前处理的有效窗口数量
                current_valid_windows = len(window_cls_list)
                
                # 计算当前批次中的最大有效窗口数量
                max_valid_windows = 0
                for batch_idx in range(batch_size):
                    valid_count = min(valid_window_counts[batch_idx].item(), current_valid_windows) 
                    max_valid_windows = max(max_valid_windows, valid_count)
                    
                print(f"Maximum valid windows in batch: {max_valid_windows}")
                
                ### Layer 2 全局聚义层 
                # 拼接输入序列
                try:
                    # 修改第535行，保留梯度
                    window_cls_list_for_global = [tensor.clone() for tensor in window_cls_list]  # 移除detach()
                    window_cls_tensor = torch.stack(window_cls_list_for_global, dim=1)
                    
                    # 为全局融合层添加专门的[CLS]和[SEP]标记
                    hidden_size = window_cls_tensor.size(2)
                    
                    # 获取第二层RoBERTa的[CLS]和[SEP]标记的嵌入
                    try:
                        # 方法1：使用嵌入矩阵获取
                        cls_token_id = torch.tensor([self.second_encoder.config.bos_token_id], device=device)
                        sep_token_id = torch.tensor([self.second_encoder.config.eos_token_id], device=device)
                        
                        # 获取嵌入
                        cls_embedding = self.second_encoder.embeddings.word_embeddings(cls_token_id).view(1, 1, hidden_size)
                        sep_embedding = self.second_encoder.embeddings.word_embeddings(sep_token_id).view(1, 1, hidden_size)
                        
                        print("Successfully obtained embeddings from embedding matrix")
                    except Exception as e1:
                        print(f"Failed to get embeddings from matrix: {str(e1)}")
                        
                        try:
                            # 方法2：使用零向量
                            cls_embedding = window_cls_tensor.mean(dim=1, keepdim=True).clone()
                            # 处理SEP
                            sep_embedding = window_cls_tensor.mean(dim=1, keepdim=True).clone()
                            print("Using mean of all window representations as CLS/SEP")
                        except Exception as e2:
                            print(f"Failed to use mean representations: {str(e2)}")
                            
                            try:
                                # 方法3：使用第一个窗口的CLS
                                cls_embedding = window_cls_tensor[:, 0:1, :].clone()
                                # 如果有多个窗口，使用最后一个窗口的CLS作为SEP
                                if window_cls_tensor.size(1) > 1:
                                    sep_embedding = window_cls_tensor[:, -1:, :].clone()
                                else:
                                    # 如果只有一个窗口，使用第一个窗口的CLS作为SEP
                                    sep_embedding = cls_embedding.clone()
                                print("Using first/last window CLS as CLS/SEP")
                            except Exception as e3:
                                print(f"Failed to use window CLS: {str(e3)}")
                                
                                # 方法4：使用微量向量
                                print("Using ones vector as fallback")
                                cls_embedding = torch.ones(1, 1, hidden_size, device=device) * 0.01  # 随机值
                                sep_embedding = torch.ones(1, 1, hidden_size, device=device) * 0.01
                    
                    # 缩放调整 - 确保特殊标记的尺度与窗口表示一致
                    window_norm = torch.norm(window_cls_tensor, dim=2).mean().clone()
                    cls_norm = torch.norm(cls_embedding).clone()
                    sep_norm = torch.norm(sep_embedding).clone()
                    
                    # 应用缩放
                    if cls_norm > 0 and sep_norm > 0:  # 避免除零错误
                        cls_embedding = cls_embedding * (window_norm / cls_norm)
                        sep_embedding = sep_embedding * (window_norm / sep_norm)
                        # 缩放特殊标记以匹配窗口表示
                    
                    # 扩展到批大小
                    cls_embedding = cls_embedding.expand(batch_size, 1, hidden_size)
                    sep_embedding = sep_embedding.expand(batch_size, 1, hidden_size)
                    
                    # 组合全局序列：[CLS] + 所有窗口表示 + [SEP]
                    # 使用克隆确保所有输入都是新副本，避免原地修改
                    cls_embedding_copy = cls_embedding.clone()
                    window_cls_tensor_copy = window_cls_tensor.clone()
                    sep_embedding_copy = sep_embedding.clone()
                    global_inputs = torch.cat([cls_embedding_copy, window_cls_tensor_copy, sep_embedding_copy], dim=1)
                    # 包含特殊标记的全局输入形状
                    
                    # 保存第二层编码器的输入CLS表示用于残差连接
                    pre_second_encoder_cls = cls_embedding_copy.squeeze(1)  # [batch_size, hidden_size]
                    
                    # 更新attention_mask，为特殊标记添加mask
                    global_attention_mask = torch.ones(batch_size, global_inputs.size(1), device=device)
                    ### 开始进行全局运算
                    if window_cls_tensor.size(1) >= 1:  # 至少有一个窗口
                        # 处理第二层编码器 - 已移除FP32转换
                        # 使用梯度检查点（当有效窗口过8个时）
                        if use_checkpointing and max_valid_windows >= 8 and 'window_cls_list' in locals() and len(window_cls_list) >= 8:
                            # 定义梯度检查点包装函数
                            def forward_second_encoder(inputs_embeds, attention_mask):
                                return self.second_encoder(
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True
                                )
                            
                            # 使用梯度检查点
                            from torch.utils.checkpoint import checkpoint
                            outputs = checkpoint(
                                forward_second_encoder, 
                                global_inputs,
                                global_attention_mask,
                                use_reentrant=False
                            )
                        else:
                            # 常规前向传播，使用当前精度
                            outputs = self.second_encoder(
                                inputs_embeds=global_inputs,
                                attention_mask=global_attention_mask,
                                output_hidden_states=True
                            )
                        
                        # 提取最终表示
                        cls_output = outputs.last_hidden_state[:, 0].clone()
                        
                        # 应用残差连接 - 将第二层编码器前后的CLS表示结合
                        alpha = 0.8  # 残差缩放因子，可调整
                        final_representation = alpha * cls_output + (1 - alpha) * pre_second_encoder_cls
                        #print(f"Applied second encoder residual connection with alpha={alpha}")
                    elif window_cls_tensor.size(1) == 0:
                        # 没有窗口的情况，这是一个意外情况，应该记录错误并采取适当措施
                        print("ERROR: No windows available for second encoder but code reached this point!")
                        # 创建一个全零向量作为备用方案
                        final_representation = torch.zeros(batch_size, self.roberta_config.hidden_size, device=device)
                        final_representation.requires_grad_(True)  # 确保梯度流动
                    else:
                        # 负数情况，这应该是不可能的，但为了健壮性
                        raise ValueError(f"Invalid window count: {window_cls_tensor.size(1)}")
                    
                    # 更新全局表示
                    cls_representation = final_representation.clone()
                    
                    # 分级内存优化策略
                    # 1. 计算当前处理的有效窗口数量
                    current_valid_windows = len(window_cls_list)
                    
                    # 2. 计算当前批次中的最大有效窗口数量
                    max_valid_windows = 0
                    for batch_idx in range(batch_size):
                        valid_count = min(valid_window_counts[batch_idx].item(), current_valid_windows) 
                        max_valid_windows = max(max_valid_windows, valid_count)
                    
                    # 3. 根据有效窗口数量决定优化策略
                    if max_valid_windows >= 8 and iteration < effective_num_repeats - 1:
                        print(f"Using EXTREME memory optimization strategy for next iteration (windows: {max_valid_windows})")
                        
                        # 启用梯度检查点模式（两个编码器均启用）
                        use_checkpointing = True
                        
                        # 清理缓存，但不分离计算图
                        import gc
                        # 不再分离窗口表示
                        # window_cls_list = [tensor.detach() for tensor in window_cls_list]  
                        
                        # 应用垃圾回收，确保释放内存
                        for _ in range(3):
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        # 测量当前显存使用情况
                        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        print(f"GPU memory: current={allocated:.2f}GB, peak={max_allocated:.2f}GB")
                        
                        # 不再分离和更新全局表示
                        # cls_representation = cls_representation_detached
                        
                    elif max_valid_windows >= 5 and iteration < effective_num_repeats - 1:
                        print(f"Using MEDIUM memory optimization strategy for next iteration (windows: {max_valid_windows})")
                        
                        # 启用梯度检查点模式（仅对第一编码器）
                        use_checkpointing = True
                        
                        # 清理缓存
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 不再更新全局表示为分离后的版本
                        # cls_representation = cls_representation_detached
                except Exception as e:
                    # 第二编码器处理出错
                    # 如果第二层处理出错，使用第一个窗口的表示
                    print(f"Error processing second encoder: {str(e)}")
                    cls_representation = window_cls_list[0]
                    # 回退到第一个窗口表示
            ### Recurrent Block End          

            #下游任务
            # 使用pooler层处理cls_representation
            pooled_output = self.pooler(cls_representation)
            logits = self.classifier(pooled_output)
            
            # 如果提供了标签，计算损失值
            loss = None
            if labels is not None:
                # Calculate loss first, and check for NaN/Inf.
                loss_fct = nn.CrossEntropyLoss()
                calculated_loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

                # Check if the calculated loss is NaN or Inf
                if torch.isnan(calculated_loss) or torch.isinf(calculated_loss):
                    # Use logger if available, otherwise print
                    print(f"WARNING: Calculated loss is NaN/Inf (likely due to problematic forward pass, e.g., 0 windows processed). Returning default high loss.")
                    # Return a default high loss value that requires grad
                    loss = torch.tensor(10.0, device=logits.device, requires_grad=True)
                else:
                    loss = calculated_loss
            
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits
            )
        
        except Exception as e:
            import traceback
            # 前向传递过程中出错
            print(f"前向传播严重错误: {str(e)}")
            print(traceback.format_exc())  # 打印完整堆栈
            # 如果前向传播出错，返回空的损失和随机初始化的logits
            dummy_logits = torch.randn(batch_size, self.num_classes, device=input_ids.device) * 0.01
            if labels is not None:
                dummy_loss = torch.tensor(10.0, device=input_ids.device, requires_grad=True)  # 高损失值表示模型失败
                return dummy_loss, dummy_logits
            else:
                # 计算loss（如果有labels）
                loss = None
                if labels is not None:
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(dummy_logits.view(-1, dummy_logits.size(-1)), labels.view(-1))
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=dummy_logits
                )
