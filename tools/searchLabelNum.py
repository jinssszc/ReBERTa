import json

labels = set()
with open('data/agnews/train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        labels.add(data['label'])

print(f"标签总数: {len(labels)}")
print(f"所有标签: {labels}")