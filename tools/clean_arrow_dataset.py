"""
Arrow文件文本清洗工具
用于清洗Arrow文件中的文本字段
使用方法：
python clean_arrow_dataset.py --input data/your_dataset.arrow --output data/cleaned_dataset.arrow --field abstract
记得修改--field中的字段关键词
上次使用：
python clean_arrow_dataset.py --input data\train\data-00005-of-00007.arrow --output data\train\cdata-00005-of-00007.arrow --field abstract
"""
from datasets import Dataset
import re
import os
import sys

def clean_text(text):
    """清洗文本，去除多余空白字符"""
    if not isinstance(text, str):
        return text
    # 去除换行和制表符，保留标点和大小写
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def clean_arrow_dataset(input_path, output_path, text_field="text"):
    """
    清洗Arrow文件中的文本字段
    
    Args:
        input_path: 输入Arrow文件路径
        output_path: 输出Arrow文件路径
        text_field: 要清洗的文本字段名
    """
    try:
        # 加载数据集
        print(f"正在加载数据集: {input_path}")
        ds = Dataset.from_file(input_path)
        
        # 检查字段是否存在
        if text_field not in ds.column_names:
            print(f"错误：字段 '{text_field}' 不存在于数据集中")
            print(f"可用字段: {ds.column_names}")
            return
        
        # 打印处理前的统计信息
        print(f"数据集大小: {len(ds)} 条记录")
        print(f"开始清洗文本字段: {text_field}")
        
        # 定义清洗函数
        def clean_example(example):
            if text_field in example and example[text_field] is not None:
                example[text_field] = clean_text(example[text_field])
            return example
        
        # 应用清洗函数
        cleaned_ds = ds.map(clean_example)
        
        # 保存清洗后的数据集
        print(f"正在保存清洗后的数据集: {output_path}")
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cleaned_ds.save_to_disk(output_path)
        
        print(f"清洗完成! 处理了 {len(cleaned_ds)} 条记录")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="清洗Arrow文件中的文本字段")
    parser.add_argument('--input', type=str, required=True, help='输入Arrow文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出Arrow文件路径')
    parser.add_argument('--field', type=str, default='text', help='要清洗的文本字段名')
    args = parser.parse_args()
    
    clean_arrow_dataset(args.input, args.output, args.field)