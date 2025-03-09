import json
import re
import difflib
from collections import defaultdict

import nltk
from nltk.corpus import words as nltk_words
from sklearn.feature_extraction.text import TfidfVectorizer

# 如果本地没有安装或下载相关资源，需要先安装/下载：
# pip install nltk scikit-learn
# nltk.download('words')

# -----------------------------------------
# 1. 参数设置
# -----------------------------------------
JSONL_FILE_PATH = 'noisyCleaner/data/test_type_15.jsonl'
TFIDF_THRESHOLD = 0.02  # TF-IDF 低于该值时可视为噪音（此值需要根据实际数据分布调试）
SIMILARITY_CUTOFF = 0.8 # 用于 difflib.get_close_matches 的相似度阈值

# -----------------------------------------
# 2. 文本预处理函数
# -----------------------------------------
def preprocess_text(text):
    """
    预处理文本：
    1. 转为小写
    2. 替换换行符为空格
    3. 去除标点符号和特殊字符
    """
    text = text.lower()
    text = re.sub(r'[\r\n]+', ' ', text)      # 将换行符替换为空格
    text = re.sub(r'[^\w\s]', '', text)       # 去除标点符号和其他非单词字符
    return text

def tokenize_text(text):
    """
    简单分词：基于空格拆分
    """
    return text.split()

# -----------------------------------------
# 3. 从 JSONL 文件中加载数据
# -----------------------------------------
all_messages = []

with open(JSONL_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        commit_msgs = data.get("commit_messages", [])
        for commit in commit_msgs:
            msg = commit.get("message", "")
            if msg:
                all_messages.append(msg)

print("从 JSONL 文件中加载的 commit messages 数量：", len(all_messages))

# -----------------------------------------
# 4. 对所有文本进行预处理，准备 TF-IDF 计算
# -----------------------------------------
processed_texts = []
for msg in all_messages:
    processed = preprocess_text(msg)
    processed_texts.append(processed)

# -----------------------------------------
# 5. 使用 scikit-learn 的 TfidfVectorizer 计算 TF-IDF
# -----------------------------------------
vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
tfidf_matrix = vectorizer.fit_transform(processed_texts)

# 得到每个词（特征）对应的索引
feature_names = vectorizer.get_feature_names_out()

# -----------------------------------------
# 6. 统计每个词的平均 TF-IDF
#    （或最大 TF-IDF、总和等不同方式）来判断其是否是噪音
# -----------------------------------------
# sum(axis=0) 得到每列（每个词）的总和
# 然后除以文档数，就得到“平均 TF-IDF”
tfidf_sum = tfidf_matrix.sum(axis=0)  # shape: (1, n_features)
n_docs = tfidf_matrix.shape[0]
tfidf_avg = (tfidf_sum / n_docs).A1   # 转成 1D array

word_tfidf_dict = {}
for i, word in enumerate(feature_names):
    word_tfidf_dict[word] = tfidf_avg[i]

# -----------------------------------------
# 7. 噪音检测：低于阈值则视为噪音
# -----------------------------------------
def is_candidate_noise_tfidf(word, tfidf_threshold):
    """
    判断某个词的平均 TF-IDF 值是否低于阈值。
    如果低于阈值，就视为噪音候选。
    """
    tfidf_val = word_tfidf_dict.get(word, 0.0)
    return (tfidf_val < tfidf_threshold)

# 这里可结合 NLTK 标准词汇库，或其他规则（如含数字等），综合判断
standard_vocab = set(nltk_words.words())

def is_candidate_noise(word):
    """
    综合 TF-IDF + 标准词汇 + 数字检测等多重规则
    """
    # 优先根据 TF-IDF 判断
    if is_candidate_noise_tfidf(word, TFIDF_THRESHOLD):
        return True
    # 或者该词中有数字
    if re.search(r'\d', word):
        return True
    # 或者它不在标准词汇表里
    if word not in standard_vocab:
        return True
    return False

# -----------------------------------------
# 8. 自动修正：使用 difflib.get_close_matches 做一个简单演示
# -----------------------------------------
def get_correction(word, standard_vocab, cutoff=0.8):
    """
    使用 difflib 根据相似度来查找可能的修正词。
    若找到相似度大于 cutoff 的匹配词，则返回第一个匹配，否则返回 None。
    """
    matches = difflib.get_close_matches(word, standard_vocab, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    return None

# -----------------------------------------
# 9. 生成修正后的文本
# -----------------------------------------
corrected_messages = []

for processed in processed_texts:
    tokens = tokenize_text(processed)
    corrected_tokens = []
    for t in tokens:
        if is_candidate_noise(t):
            suggestion = get_correction(t, standard_vocab, cutoff=SIMILARITY_CUTOFF)
            if suggestion:
                corrected_tokens.append(suggestion)
            else:
                corrected_tokens.append(t)  # 找不到合适的就先保留
        else:
            corrected_tokens.append(t)
    corrected_msg = " ".join(corrected_tokens)
    corrected_messages.append(corrected_msg)

# -----------------------------------------
# 10. 对比输出若干条修正结果
# -----------------------------------------
print("\n[修正前 -> 修正后] 示例：")
num_show = min(5, len(processed_texts))
for i in range(num_show):
    print(f"原文: {processed_texts[i]}")
    print(f"修正: {corrected_messages[i]}")
    print("-"*60)
