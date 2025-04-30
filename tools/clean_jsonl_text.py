import json
import re
import sys

def clean_text(text):
    # 去除换行和制表符，保留标点和大小写
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# 主处理流程
def clean_jsonl(input_path, output_path, text_field="text"):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if text_field in obj and isinstance(obj[text_field], str):
                    obj[text_field] = clean_text(obj[text_field])
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"跳过无法解析的行: {e}", file=sys.stderr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="清理JSONL文件中的文本字段")
    parser.add_argument('--input', type=str, required=True, help='输入jsonl路径')
    parser.add_argument('--output', type=str, required=True, help='输出jsonl路径')
    parser.add_argument('--field', type=str, default='text', help='要清理的字段名')
    args = parser.parse_args()
    clean_jsonl(args.input, args.output, args.field)
