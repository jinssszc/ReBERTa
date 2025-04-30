from datasets import Dataset
import os

data_dir = "data/train"
label_set = set()

for fname in os.listdir(data_dir):
    if fname.endswith(".arrow"):
        ds = Dataset.from_file(os.path.join(data_dir, fname))
        for label in ds["descriptor"]:
            if label is not None:
                label_set.add(label)

print(f"标签总数: {len(label_set)}")
print(f"部分标签示例: {list(label_set)[:10]}")