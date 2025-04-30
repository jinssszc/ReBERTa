import json
import random
import argparse
import os

def split_jsonl(input_path, train_path, val_path, split_ratio=0.9, seed=42):
    """
    将 input_path 下的 JSONL 文件按 split_ratio 随机划分为 train_path 和 val_path。
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"读取样本数: {len(lines)}")
    random.seed(seed)
    random.shuffle(lines)
    split_idx = int(len(lines) * split_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    print(f"训练集: {len(train_lines)}，验证集: {len(val_lines)}")
    with open(train_path, 'w', encoding='utf-8') as f_train:
        f_train.writelines(train_lines)
    with open(val_path, 'w', encoding='utf-8') as f_val:
        f_val.writelines(val_lines)
    print(f"已写入 {train_path} 和 {val_path}")

def main():
    parser = argparse.ArgumentParser(description="Split JSONL dataset into train and validation sets.")
    parser.add_argument('--input', type=str, default='data/arxiv-clf/row.jsonl', help='输入 JSONL 文件路径')
    parser.add_argument('--train', type=str, default='data/arxiv-clf/train.jsonl', help='输出训练集文件路径')
    parser.add_argument('--val', type=str, default='data/arxiv-clf/validation.jsonl', help='输出验证集文件路径')
    parser.add_argument('--ratio', type=float, default=0.9, help='训练集比例 (0-1)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.train), exist_ok=True)
    os.makedirs(os.path.dirname(args.val), exist_ok=True)
    split_jsonl(args.input, args.train, args.val, args.ratio, args.seed)

if __name__ == '__main__':
    main()
