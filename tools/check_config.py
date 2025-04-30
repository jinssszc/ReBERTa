# -*- coding: utf-8 -*-
import torch
import json
from pathlib import Path

def check_model_config(model_path):
    """检查模型配置文件的内容"""
    print(f"\n正在检查模型文件: {model_path}")
    print("="*80)
    
    # 加载模型文件
    from torch.serialization import safe_globals
    import numpy
    with safe_globals([numpy._core.multiarray.scalar, numpy.dtype]):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # 检查基本结构
    print("\n1. 检查文件基本结构:")
    print("-"*40)
    for key in ckpt.keys():
        print(f"找到键: {key}")
    
    # 检查配置信息
    print("\n2. 检查配置信息:")
    print("-"*40)
    config = ckpt.get("config", {})
    if not config:
        print("警告: 未找到配置信息!")
    else:
        print("配置包含以下键:")
        for key in config.keys():
            print(f"- {key}")
            
        # 特别检查标签映射
        print("\n3. 检查标签映射:")
        print("-"*40)
        label2id = config.get("label2id")
        id2label = config.get("id2label")
        
        if label2id:
            print("label2id 映射存在:")
            print(json.dumps(label2id, indent=2, ensure_ascii=False))
        else:
            print("警告: 未找到 label2id 映射!")
            
        if id2label:
            print("\nid2label 映射存在:")
            print(json.dumps(id2label, indent=2, ensure_ascii=False))
        else:
            print("警告: 未找到 id2label 映射!")

if __name__ == "__main__":
    # 检查最佳准确率模型
    model_path = "models/saved/best_accuracy_model.pt" 
    if Path(model_path).exists():
        check_model_config(model_path)
    else:
        print(f"错误: 未找到模型文件 {model_path}")
