import json
import re
from collections import Counter

# 读取数据集并统计词频
word_counter = Counter()

# 正则表达式用于匹配单词
word_pattern = re.compile(r'\b\w+\b')

# with open('BART/getGitData/rnsum.jsonl', 'r') as file:
#     for line in file:
#         data = json.loads(line)
#         commit_messages = data.get("commit_messages", "")  # 假设每条记录的说明字段为 "release_note"
#         for commit_message in commit_messages:
#             words = word_pattern.findall(commit_message.lower())
#             word_counter.update(words)
with open('BART/getGitData/rnsum.jsonl', 'r') as file:
    for line in file:
        release_notes = line.strip().lower()
        for release_note in release_notes:
            words = word_pattern.findall(release_note)
            word_counter.update(words)

# 获取出现次数最高的前300个词
most_common_words = word_counter.most_common(300)

# 打印结果
print("Top 30 most common words:")
for word, count in most_common_words:
    print(f"{word}: {count}")