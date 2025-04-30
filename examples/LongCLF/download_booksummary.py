import os
import urllib.request
import tarfile
import subprocess
import sys
import shutil

# 创建数据目录
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
booksummaries_dir = os.path.join(data_dir, "booksummaries")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(booksummaries_dir, exist_ok=True)

# 下载数据集
dataset_url = "http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
tar_file_path = os.path.join(booksummaries_dir, "booksummaries.tar.gz")

if not os.path.exists(tar_file_path):
    print(f"正在下载 BookSummaries 数据集...")
    urllib.request.urlretrieve(dataset_url, tar_file_path)
    print(f"下载完成: {tar_file_path}")

# 解压数据集
if os.path.exists(tar_file_path) and not os.path.exists(os.path.join(booksummaries_dir, "booksummaries.txt")):
    print(f"正在解压数据集...")
    with tarfile.open(tar_file_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    
    # 检查和移动文件（如果需要）
    extracted_file = os.path.join(data_dir, "booksummaries.txt")
    target_file = os.path.join(booksummaries_dir, "booksummaries.txt")
    
    if os.path.exists(extracted_file) and not os.path.exists(target_file):
        shutil.move(extracted_file, target_file)
        print(f"文件已移动到: {target_file}")
    
    print(f"解压完成")

# 验证文件存在
booksummaries_file = os.path.join(booksummaries_dir, "booksummaries.txt")
if not os.path.exists(booksummaries_file):
    print(f"错误: 找不到数据文件 {booksummaries_file}")
    print(f"当前目录内容: {os.listdir(booksummaries_dir)}")
    sys.exit(1)
else:
    print(f"找到数据文件: {booksummaries_file}")

# 确保必要的模块已安装
try:
    import pandas as pd
    import json
except ImportError:
    print("安装必要的依赖...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd
    import json

# 预处理脚本的修改版本
print("正在预处理BookSummaries数据集...")

def parse_json_column(genre_data):
    try:
        return json.loads(genre_data)
    except Exception as e:
        return None

def load_booksummaries_data(book_path):
    print(f"正在加载数据: {book_path}")
    book_df = pd.read_csv(book_path, sep='\t', names=["Wikipedia article ID",
                                                     "Freebase ID",
                                                     "Book title",
                                                     "Author",
                                                     "Publication date",
                                                     "genres",
                                                     "summary"],
                         converters={'genres': parse_json_column})
    book_df = book_df.dropna(subset=['genres', 'summary'])
    book_df['word_count'] = book_df['summary'].str.split().str.len()
    book_df = book_df[book_df['word_count'] >= 10]
    train = book_df.sample(frac=0.8, random_state=22)
    rest = book_df.drop(train.index)
    dev = rest.sample(frac=0.5, random_state=22)
    test = rest.drop(dev.index)
    return train, dev, test

# 单本处理模式
text_set = {'train': [], 'dev': [], 'test': []}
label_set = {'train': [], 'dev': [], 'test': []}
train, dev, test = load_booksummaries_data(booksummaries_file)

print('处理训练集:', len(train))
text_set['train'] = train['summary'].tolist()
train_genres = train['genres'].tolist()
label_set['train'] = [list(genre.values()) for genre in train_genres if genre]

print('处理开发集:', len(dev))
text_set['dev'] = dev['summary'].tolist()
dev_genres = dev['genres'].tolist()
label_set['dev'] = [list(genre.values()) for genre in dev_genres if genre]

print('处理测试集:', len(test))
text_set['test'] = test['summary'].tolist()
test_genres = test['genres'].tolist()
label_set['test'] = [list(genre.values()) for genre in test_genres if genre]

# 保存处理后的数据
for split in ['train', 'dev', 'test']:
    texts = text_set[split]
    labels = label_set[split]
    assert len(texts) == len(labels)
    print(f'{split} 数据集大小:', len(texts))
    output_file = os.path.join(booksummaries_dir, f'{split}.jsonl')
    with open(output_file, 'w', encoding='utf-8') as writer:
        for text, label in zip(texts, labels):
            writer.writelines(json.dumps({'text': text, 'label': label}, ensure_ascii=False) + '\n')
    print(f'已保存到: {output_file}')

print("预处理完成!")
