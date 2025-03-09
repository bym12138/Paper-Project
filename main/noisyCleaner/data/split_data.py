# import math

# def split_jsonl_file(input_file_path, output_prefix, num_splits):
#     # 计算文件总行数
#     with open(input_file_path, 'r') as file:
#         total_lines = sum(1 for line in file)

#     # 计算每个文件的行数
#     lines_per_split = math.ceil(total_lines / num_splits)

#     # 开始分割文件
#     current_split = 1
#     current_line = 0
#     output_file = None

#     with open(input_file_path, 'r') as file:
#         for line in file:
#             if current_line % lines_per_split == 0:
#                 if output_file:
#                     output_file.close()
#                 output_file_name = f"{output_prefix}_part{current_split}.jsonl"
#                 output_file = open(output_file_name, 'w')
#                 current_split += 1
#             output_file.write(line)
#             current_line += 1

#         if output_file:
#             output_file.close()

#     print(f"文件分割完成，生成了{current_split - 1}个文件，总行数为{total_lines}")

# # 调用函数，假设文件名为 'train_type_f.jsonl'，输出文件前缀为 'split_train_type_f'
# split_jsonl_file('noisyCleaner/data/train_type.jsonl', 'noisyCleaner/data/train_type', 30)


import json
import random
from sklearn.model_selection import train_test_split

# 设置随机种子以确保可重复性
random.seed(42)

# 读取数据并处理解析错误
data = []
with open('noisyCleaner/data/train_type_score_copy.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.decoder.JSONDecodeError as e:
            print(f"解析错误: {e}，在行内容: {line}")

# 打乱数据
random.shuffle(data)

# 切分数据集为训练集、剩余部分（验证集+测试集）
train_data, remaining_data = train_test_split(data, train_size=0.7, random_state=42)

# 保存数据集到JSONL文件
def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

save_jsonl(train_data, 'noisyCleaner/data/train_type_score.jsonl')
save_jsonl(remaining_data, 'noisyCleaner/data/test_type_score.jsonl')
