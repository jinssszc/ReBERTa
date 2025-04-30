"""
Arrow数据集结构分析器
用于检查Arrow文件中的特征和标签字段
main函数里的data_dir指向arrow所在的文件夹
"""
from datasets import Dataset
import os
import pandas as pd
from collections import Counter
import json

def analyze_arrow_dataset(data_dir, sample_size=5):
    """分析目录中的所有Arrow文件"""
    print(f"正在分析目录: {data_dir} 中的Arrow文件")
    
    # 确保目录存在
    if not os.path.exists(data_dir):
        print(f"错误: 目录 {data_dir} 不存在")
        return
    
    # 查找所有arrow文件
    arrow_files = [f for f in os.listdir(data_dir) if f.endswith('.arrow')]
    if not arrow_files:
        print(f"未找到.arrow文件在 {data_dir}")
        return
    
    print(f"找到 {len(arrow_files)} 个Arrow文件")
    
    # 分析每个文件
    for idx, fname in enumerate(arrow_files):
        file_path = os.path.join(data_dir, fname)
        print(f"\n[{idx+1}/{len(arrow_files)}] 分析文件: {fname}")
        
        try:
            # 加载数据集
            ds = Dataset.from_file(file_path)
            
            # 基本信息
            print(f"数据集大小: {len(ds)} 条记录")
            print(f"特征字段: {ds.column_names}")
            
            # 详细分析每个字段
            for column in ds.column_names:
                print(f"\n字段: {column}")
                
                # 获取字段数据类型
                field_type = type(ds[0][column]).__name__ if len(ds) > 0 and ds[0][column] is not None else "未知"
                print(f"数据类型: {field_type}")
                
                # 样本值
                if len(ds) > 0:
                    samples = ds[column][:min(sample_size, len(ds))]
                    print(f"示例值: {samples}")
                
                # 检查是否可能是标签字段
                if column.lower() in ['label', 'labels', 'class', 'classes', 'category', 'categories', 
                                     'target', 'descriptor', 'tag', 'tags']:
                    # 统计唯一值
                    valid_values = [v for v in ds[column] if v is not None]
                    unique_values = set(valid_values)
                    
                    print(f"可能的标签字段! 包含 {len(unique_values)} 个唯一值")
                    
                    # 显示所有唯一标签
                    if len(unique_values) < 100:  # 避免显示太多标签
                        print(f"唯一值: {sorted(unique_values)}")
                    else:
                        print(f"唯一值数量过多，仅显示前10个: {sorted(list(unique_values))[:10]}")
                    
                    # 显示标签分布
                    counter = Counter(valid_values)
                    print(f"标签分布: {dict(counter.most_common(5))}")
                
                # 检查文本字段
                if field_type in ['str', 'string'] or column.lower() in ['text', 'content', 'document', 'article', 'abstract']:
                    # 检查非空文本的数量
                    non_empty = sum(1 for t in ds[column] if t and isinstance(t, str) and t.strip())
                    print(f"非空文本数量: {non_empty}/{len(ds)} ({non_empty/len(ds)*100:.2f}%)")
                    
                    # 计算平均长度
                    if non_empty > 0:
                        avg_len = sum(len(t) for t in ds[column] if t and isinstance(t, str)) / non_empty
                        print(f"文本平均长度: {avg_len:.2f} 字符")
            
            # 尝试将第一条记录转换为字典显示
            if len(ds) > 0:
                print("\n第一条记录结构:")
                try:
                    record = dict(ds[0])
                    # 对于长文本，进行截断
                    for k, v in record.items():
                        if isinstance(v, str) and len(v) > 500:
                            record[k] = v[:500] + "..."
                    print(json.dumps(record, ensure_ascii=False, indent=2))
                except:
                    print("无法以JSON格式显示第一条记录")
        
        except Exception as e:
            print(f"分析文件 {fname} 时出错: {str(e)}")

if __name__ == "__main__":
    # 可以修改为您的数据路径
    data_dir = "data/train"
    analyze_arrow_dataset(data_dir)