#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载和处理hyperpartisan数据集的脚本
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import subprocess
import argparse


def download_file(url, destination):
    """
    下载文件并显示进度条
    
    Args:
        url (str): 下载URL
        destination (str): 目标文件路径
    """
    print(f"下载 {url} 到 {destination}")
    
    def report_progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r下载进度: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, reporthook=report_progress)
    print()  # 换行


def main():
    parser = argparse.ArgumentParser(description="下载并处理hyperpartisan数据集")
    parser.add_argument('--data-dir', type=str, default='data/hyperpartisan',
                        help='数据存储目录')
    args = parser.parse_args()
    
    # 确保目录存在
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    # 文件URL
    articles_url = "https://zenodo.org/record/1489920/files/articles-training-byarticle-20181122.zip"
    labels_url = "https://zenodo.org/record/1489920/files/ground-truth-training-byarticle-20181122.zip"
    splits_url = "https://raw.githubusercontent.com/allenai/longformer/master/scripts/hp-splits.json"
    
    # 下载和提取文件
    articles_zip = os.path.join(data_dir, "articles-training-byarticle-20181122.zip")
    labels_zip = os.path.join(data_dir, "ground-truth-training-byarticle-20181122.zip")
    splits_file = os.path.join(data_dir, "hp-splits.json")
    
    if not os.path.exists(articles_zip):
        download_file(articles_url, articles_zip)
    
    if not os.path.exists(labels_zip):
        download_file(labels_url, labels_zip)
    
    if not os.path.exists(splits_file):
        download_file(splits_url, splits_file)
    
    # 解压缩文件
    articles_xml = os.path.join(data_dir, "articles-training-byarticle-20181122.xml")
    labels_xml = os.path.join(data_dir, "ground-truth-training-byarticle-20181122.xml")
    
    if not os.path.exists(articles_xml):
        print(f"解压缩 {articles_zip}")
        with zipfile.ZipFile(articles_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    if not os.path.exists(labels_xml):
        print(f"解压缩 {labels_zip}")
        with zipfile.ZipFile(labels_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    # 运行预处理脚本
    print("处理数据集...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocess_script = os.path.join(current_dir, "hyperpartisan_preprocess.py")
    
    cmd = [
        sys.executable,
        preprocess_script,
        "--articles-file", articles_xml,
        "--labels-file", labels_xml,
        "--output-dir", data_dir
    ]
    
    subprocess.run(cmd, check=True)
    
    # 删除不需要的zip文件（可选）
    if os.path.exists(articles_zip):
        os.remove(articles_zip)
        print(f"已删除 {articles_zip}")
    
    if os.path.exists(labels_zip):
        os.remove(labels_zip)
        print(f"已删除 {labels_zip}")
    
    print("完成！数据集已下载和处理为jsonl格式。")
    print(f"数据存储在 {data_dir} 目录中")


if __name__ == "__main__":
    main()
