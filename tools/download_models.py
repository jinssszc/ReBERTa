"""
下载RoBERTa模型和分词器到本地目录
"""

import os
import argparse
import torch
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from pathlib import Path

def download_roberta(output_dir="models/pretrained"):
    """
    下载RoBERTa-base模型和分词器到指定目录
    
    Args:
        output_dir (str): 保存路径
    """
    print(f"开始下载RoBERTa-base模型和分词器到 {output_dir}")
    
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载和保存模型配置
    print("下载模型配置...")
    config = RobertaConfig.from_pretrained('roberta-base')
    config.save_pretrained(output_dir)
    print(f"模型配置已保存到 {output_dir}")
    
    # 下载和保存分词器
    print("下载分词器...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.save_pretrained(output_dir)
    print(f"分词器已保存到 {output_dir}")
    
    # 下载和保存模型权重
    print("下载模型权重...")
    model = RobertaModel.from_pretrained('roberta-base')
    model.save_pretrained(output_dir)
    print(f"模型权重已保存到 {output_dir}")
    
    # 显示保存的文件
    print("\n已下载以下文件:")
    for file in Path(output_dir).glob("*"):
        print(f" - {file.name} ({file.stat().st_size / 1024 / 1024:.2f} MB)")
    
    print("\n下载完成! 您现在可以在代码中使用本地模型:")
    print(f"  RobertaModel.from_pretrained('{output_dir}')")
    print(f"  RobertaTokenizer.from_pretrained('{output_dir}')")

def main():
    parser = argparse.ArgumentParser(description="下载RoBERTa模型和分词器")
    parser.add_argument("--output_dir", type=str, default="models/pretrained", 
                        help="模型和分词器保存路径")
    args = parser.parse_args()
    
    download_roberta(args.output_dir)

if __name__ == "__main__":
    main()
