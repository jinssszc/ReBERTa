import os
import json
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
from sklearn.model_selection import train_test_split
from langdetect import detect, LangDetectException


def parse_hyperpartisan_xml(articles_file, labels_file):
    """
    解析hyperpartisan数据集的XML文件
    
    Args:
        articles_file (str): 文章XML文件路径
        labels_file (str): 标签XML文件路径
        
    Returns:
        DataFrame: 包含文章id、文章内容和标签的数据框
    """
    # 解析标签文件
    tree = ET.parse(labels_file)
    root = tree.getroot()
    
    # 创建文章id到标签的映射
    label_dict = {}
    for article in root.findall('.//article'):
        article_id = article.get('id')
        # 将hyperpartisan属性转换为布尔值，然后转换为字符串"true"或"false"
        hyperpartisan = str(article.get('hyperpartisan')).lower()
        if hyperpartisan == "true":
            label = "true"
        else:
            label = "false"
        label_dict[article_id] = label
    
    # 解析文章文件
    tree = ET.parse(articles_file)
    root = tree.getroot()
    
    # 提取文章内容
    articles = []
    for article in root.findall('.//article'):
        article_id = article.get('id')
        if article_id in label_dict:
            # 提取文章文本
            title = article.find('title')
            title_text = title.text if title is not None and title.text else ""
            
            # 收集所有段落
            paragraphs = article.findall('.//p')
            content = " ".join([p.text for p in paragraphs if p.text])
            
            # 合并标题和内容
            text = title_text + " " + content
            text = text.strip()
            
            # 添加到列表
            articles.append({
                'id': article_id,
                'text': text,
                'label': label_dict[article_id]
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(articles)
    return df


def clean_text(text):
    """
    清理文本，去除多余的空格和换行符
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 清理后的文本
    """
    # 替换换行符和制表符
    text = text.replace('\n', ' ').replace('\t', ' ')
    # 替换多个空格为单个空格
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


def is_english(text):
    """
    检测文本是否为英文
    
    Args:
        text (str): 输入文本
        
    Returns:
        bool: 如果是英文返回True，否则返回False
    """
    try:
        # 只检测前1000个字符，提高速度
        lang = detect(text[:1000])
        return lang == 'en'
    except LangDetectException:
        # 如果检测失败，默认返回False
        return False


def process_hyperpartisan_dataset(articles_file, labels_file, output_dir):
    """
    处理hyperpartisan数据集并保存为jsonl格式
    
    Args:
        articles_file (str): 文章XML文件路径
        labels_file (str): 标签XML文件路径
        output_dir (str): 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析数据
    df = parse_hyperpartisan_xml(articles_file, labels_file)
    
    # 清理文本
    df['text'] = df['text'].apply(clean_text)
    
    # 删除text为空的行
    df = df[df['text'].str.len() > 0].reset_index(drop=True)
    
    # 过滤非英文样本
    print("检测并过滤非英文样本...")
    english_mask = df['text'].apply(is_english)
    non_english_count = (~english_mask).sum()
    df = df[english_mask].reset_index(drop=True)
    print(f"已过滤 {non_english_count} 个非英文样本，剩余 {len(df)} 个英文样本")
    
    # 划分数据集
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    # 保存为jsonl文件
    for split_name, split_df in [('train', train_df), ('dev', val_df), ('test', test_df)]:
        output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in split_df.iterrows():
                json_obj = {'text': row['text'], 'label': row['label']}
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        print(f"已保存 {split_name}.jsonl")


def main():
    parser = argparse.ArgumentParser(description="处理hyperpartisan数据集")
    parser.add_argument('--articles-file', type=str, required=True, 
                        help='文章XML文件路径')
    parser.add_argument('--labels-file', type=str, required=True, 
                        help='标签XML文件路径')
    parser.add_argument('--output-dir', type=str, default='data/hyperpartisan',
                        help='输出目录')
    
    args = parser.parse_args()
    
    process_hyperpartisan_dataset(args.articles_file, args.labels_file, args.output_dir)


if __name__ == "__main__":
    main()
