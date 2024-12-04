import json
import random
from sklearn.model_selection import train_test_split

# 设置随机种子以确保可重复性
random.seed(42)

# 读取数据
with open('/data/home2/kjb/bym/project/BART/getGitData/rnsum.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

# 打乱数据
random.shuffle(data)

# 切分数据集为训练集、剩余部分（验证集+测试集）
train_data, remaining_data = train_test_split(data, train_size=0.7, random_state=42)

# 将剩余部分切分为验证集和测试集
val_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=42)

# 保存数据集到JSONL文件
def save_jsonl(data, filename):
    with open(filename, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

save_jsonl(train_data, '/data/home2/kjb/bym/project/BART/getGitData/train.jsonl')
save_jsonl(val_data, '/data/home2/kjb/bym/project/BART/getGitData/val.jsonl')
save_jsonl(test_data, '/data/home2/kjb/bym/project/BART/getGitData/test.jsonl')