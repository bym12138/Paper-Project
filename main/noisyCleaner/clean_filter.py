import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import FastText
import json
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# 下载NLTK的必要数据
nltk.download('punkt')
nltk.download('stopwords')


class ReleaseNoteCleaner:
    def __init__(self, commit_messages, release_notes):
        """
        初始化ReleaseNoteCleaner类。

        :param commit_messages: List of commit messages.
        :param release_notes: Dictionary of release notes with categories as keys and list of notes as values.
        """
        self.commit_messages = commit_messages
        self.release_notes = release_notes
        self.replacement_rules = {}
        self.cleaned_commit_messages = []
        self.cleaned_release_notes = {}

    def _flatten_release_notes(self, release_notes):
        """
        将嵌套的release_notes转换为平面列表。

        :param release_notes: Dictionary of release notes.
        :return: List of release notes.
        """
        flattened = []
        for key, notes in release_notes.items():
            for note in notes:
                flattened.append(note)
        return flattened

    def _clean_message(self, message):
        """
        清洗单条消息。

        :param message: 原始消息字符串。
        :return: 清洗后的消息字符串。
        """
        # 去除网址链接
        message = re.sub(r'http\S+|www\S+|https\S+', '', message, flags=re.MULTILINE)
        # 去除哈希值（假设哈希值为连续的16进制字符）
        message = re.sub(r'\b[a-fA-F0-9]{16,}\b', '', message)
        # 去除无意义符号（保留字母、数字和空格）
        message = re.sub(r'[^\w\s]', '', message)
        # 转换为小写
        message = message.lower()
        # 去除多余的空格
        message = re.sub(r'\s+', ' ', message).strip()
        return message if message else None

    def clean(self):
        """
        清洗commit_messages和release_notes。

        :return: None
        """
        # 清洗commit_messages
        self.cleaned_commit_messages = []
        for message in self.commit_messages:
            cleaned_message = self._clean_message(message)
            if cleaned_message:
                self.cleaned_commit_messages.append(cleaned_message)

        # 清洗release_notes
        self.cleaned_release_notes = {}
        for category, notes in self.release_notes.items():
            self.cleaned_release_notes[category] = []
            for note in notes:
                cleaned_note = self._clean_message(note)
                if cleaned_note:
                    self.cleaned_release_notes[category].append(cleaned_note)

    def statistical_analysis(self, texts, threshold=5):
        """
        统计分析，计算词频和TF-IDF。

        :param texts: List of文本。
        :param threshold: 词频阈值。
        :return: word_counter, feature_names, tfidf_matrix
        """
        word_counter = Counter()
        for text in texts:
            words = nltk.word_tokenize(text)
            word_counter.update(words)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        return word_counter, feature_names, tfidf_matrix

    def build_dynamic_lexicon(self, texts, vector_size=100, window=5, min_count=1, workers=4):
        """
        构建动态词汇表，训练FastText模型。

        :param texts: List of文本。
        :param vector_size: FastText向量大小。
        :param window: FastText窗口大小。
        :param min_count: 最小词频。
        :param workers: FastText工作线程数。
        :return: vocab, model
        """
        tokenized_texts = [nltk.word_tokenize(text) for text in texts]
        model = FastText(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        vocab = set()
        for text in texts:
            words = nltk.word_tokenize(text)
            vocab.update(words)
        return vocab, model

    def update_rules(self, word_counter, vocab, model, frequency_threshold=2):
        """
        基于词频和相似度更新替换规则。

        :param word_counter: 词频计数器。
        :param vocab: 词汇表。
        :param model: FastText模型。
        :param frequency_threshold: 词频阈值。
        :return: replacement_rules
        """
        replacement_rules = {}
        for word, count in word_counter.items():
            if count < frequency_threshold and word in vocab:
                similar_words = model.wv.most_similar(word, topn=1)
                if similar_words:
                    similar_word, similarity = similar_words[0]
                    if similarity > 0.7:
                        replacement_rules[word] = similar_word
        return replacement_rules

    def cluster_and_detect_noise(self, vocab, model, min_cluster_size=2, min_samples=1):
        """
        使用HDBSCAN聚类，检测噪音词。

        :param vocab: 词汇表。
        :param model: FastText模型。
        :param min_cluster_size: HDBSCAN最小聚类大小。
        :param min_samples: HDBSCAN最小样本数。
        :return: normal_words, noise_words, word_vectors, cluster_labels
        """
        word_vectors = []
        words = []
        for word in vocab:
            if word in model.wv:
                word_vectors.append(model.wv[word])
                words.append(word)

        word_vectors = np.array(word_vectors)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
        cluster_labels = clusterer.fit_predict(word_vectors)

        normal_words = []
        noise_words = []
        for word, label in zip(words, cluster_labels):
            if label == -1:
                noise_words.append(word)
            else:
                normal_words.append(word)

        return normal_words, noise_words, word_vectors, cluster_labels

    def generate_rules_from_noise(self, noise_words, normal_words, model, similarity_threshold=0.7):
        """
        基于噪音词生成替换规则。

        :param noise_words: 噪音词列表。
        :param normal_words: 正常词列表。
        :param model: FastText模型。
        :param similarity_threshold: 相似度阈值。
        :return: replacement_rules
        """
        replacement_rules = {}
        for noise_word in noise_words:
            similar_words = model.wv.most_similar(noise_word, topn=5)
            for similar_word, similarity in similar_words:
                if similar_word in normal_words and similarity >= similarity_threshold:
                    replacement_rules[noise_word] = similar_word
                    break
        return replacement_rules

    def apply_replacement(self, text, rules):
        """
        应用替换规则到文本。

        :param text: 原始文本。
        :param rules: 替换规则字典。
        :return: 替换后的文本。
        """
        for wrong, correct in rules.items():
            text = re.sub(rf'\b{re.escape(wrong)}\b', correct, text)
        return text

    def plot_noise_distribution(self, word_vectors, cluster_labels, output_path='noise_distribution.png'):
        """
        可视化噪音分布的聚类图。

        :param word_vectors: 词向量数组。
        :param cluster_labels: 聚类标签。
        :param output_path: 输出图片路径。
        :return: None
        """
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(word_vectors)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=cluster_labels, palette='Set1', legend='full')
        plt.title('Noise Distribution Clustering')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

    def dynamic_filter(self, plot=False):
        """
        执行动态过滤，生成替换规则并应用到文本中。

        :param plot: 是否生成噪音分布图。
        :return: cleaned_commit_messages, cleaned_release_notes
        """
        # 清洗文本
        self.clean()

        # 准备清洗后的所有文本用于统计分析和词汇表构建
        cleaned_release_notes_flat = self._flatten_release_notes(self.cleaned_release_notes)
        cleaned_all_texts = self.cleaned_commit_messages + cleaned_release_notes_flat

        # 统计分析
        word_counter, feature_names, tfidf_matrix = self.statistical_analysis(cleaned_all_texts)
        print("\n词频统计（前10）：")
        for word, count in word_counter.most_common(10):
            print(f"{word}: {count}")

        # 动态词汇表构建
        vocab, model = self.build_dynamic_lexicon(cleaned_all_texts)
        print(f"\n动态词汇表大小：{len(vocab)}")

        # 动态规则更新（基于词频）
        frequency_replacement_rules = self.update_rules(word_counter, vocab, model)
        print("\n动态替换规则（基于词频）：")
        for wrong, correct in frequency_replacement_rules.items():
            print(f"'{wrong}' -> '{correct}'")

        # 聚类与噪音检测
        normal_words, noise_words, word_vectors, cluster_labels = self.cluster_and_detect_noise(vocab, model)
        print(f"\n检测到的噪音词数量：{len(noise_words)}")
        print("\n检测到的噪音词：")
        print(noise_words)

        # 基于聚类生成替换规则
        cluster_replacement_rules = self.generate_rules_from_noise(noise_words, normal_words, model)
        print("\n动态替换规则（基于聚类）：")
        for wrong, correct in cluster_replacement_rules.items():
            print(f"'{wrong}' -> '{correct}'")

        # 合并所有替换规则
        self.replacement_rules = {**frequency_replacement_rules, **cluster_replacement_rules}
        print(f"\n总共生成了 {len(self.replacement_rules)} 条替换规则。")

        # 应用所有替换规则到commit_messages
        self.cleaned_commit_messages = [self.apply_replacement(text, self.replacement_rules) for text in self.cleaned_commit_messages]

        # 应用所有替换规则到release_notes
        for category, notes in self.cleaned_release_notes.items():
            self.cleaned_release_notes[category] = [self.apply_replacement(note, self.replacement_rules) for note in notes]

        # 可视化噪音分布
        if plot:
            self.plot_noise_distribution(word_vectors, cluster_labels)

        return self.cleaned_commit_messages, self.cleaned_release_notes


def process_jsonl(file_path):
    """
    读取JSONL文件，提取commit_messages和release_notes。

    :param file_path: JSONL文件路径。
    :return: commit_messages, release_notes
    """
    commit_messages = []
    release_notes = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 提取commit_messages
            commits = data.get('commit_messages', [])
            commit_messages.extend(commits)

            # 提取release_notes
            release_note = data.get('release_note', {})
            for category, notes in release_note.items():
                if category not in release_notes:
                    release_notes[category] = []
                release_notes[category].extend(notes)

    return commit_messages, release_notes


def save_cleaned_jsonl(input_file, output_file, cleaned_commits, cleaned_release_notes):
    """
    读取原始JSONL文件，应用清洗后的数据，并保存到新的JSONL文件。

    :param input_file: 原始JSONL文件路径。
    :param output_file: 清洗后JSONL文件路径。
    :param cleaned_commits: 清洗后的commit_messages列表。
    :param cleaned_release_notes: 清洗后的release_notes字典。
    :return: None
    """
    print("保存清洗后的数据到新的JSONL文件...")
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(tqdm(f_in, desc="写入清洗数据")):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # 如果行不是有效的JSON，跳过
                continue

            # 仅保留 'commit_messages' 和 'release_note'
            cleaned_entry = {}

            # 清洗commit_messages
            commits = data.get('commit_messages', [])
            cleaned_commits_entry = []
            for i, commit in enumerate(commits):
                if i < len(cleaned_commits):
                    cleaned_commits_entry.append(cleaned_commits[i])
                else:
                    # 如果有缺失，保留原始
                    cleaned_commits_entry.append(commit)
            cleaned_entry['commit_messages'] = cleaned_commits_entry

            # 清洗release_notes
            release_note = data.get('release_note', {})
            cleaned_release_note = {}
            for category, notes in release_note.items():
                cleaned_notes = []
                for i, note in enumerate(notes):
                    if category in cleaned_release_notes and i < len(cleaned_release_notes[category]):
                        cleaned_notes.append(cleaned_release_notes[category][i])
                    else:
                        # 如果有缺失，保留原始
                        cleaned_notes.append(note)
                cleaned_release_note[category] = cleaned_notes
            cleaned_entry['release_note'] = cleaned_release_note

            # 写入清洗后的数据
            f_out.write(json.dumps(cleaned_entry, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    input_file = 'noisyCleaner/data/test_ori.jsonl'
    output_file = 'noisyCleaner/data/train_min_f.jsonl'

    print("读取和处理JSONL文件...")
    commit_messages, release_notes = process_jsonl(input_file)
    all_texts = commit_messages + [note for notes in release_notes.values() for note in notes]
    print(f"总共有 {len(all_texts)} 条文本进行处理。")

    cleaner = ReleaseNoteCleaner(commit_messages, release_notes)
    cleaned_commits, cleaned_release_notes = cleaner.dynamic_filter(plot=True)

    # print("\n清洗后的commit_messages：")
    # for message in cleaned_commits:
    #     print(message)

    # print("\n清洗后的release_notes：")
    # for category, notes in cleaned_release_notes.items():
    #     print(f"{category}:")
    #     for note in notes:
    #         print(f"  - {note}")

    print("\n应用清洗后的数据并保存到新的JSONL文件...")
    save_cleaned_jsonl(input_file, output_file, cleaned_commits, cleaned_release_notes)
    print(f"\n清洗后的数据已保存到 {output_file}")









# import re
# import nltk
# from collections import defaultdict, Counter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models import FastText
# import json
# import hdbscan
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # 下载NLTK的必要数据
# nltk.download('punkt')
# nltk.download('stopwords')


# class ReleaseNoteCleaner:
#     def __init__(self, commit_messages):
#         self.commit_messages = commit_messages

#     def clean(self):
#         self.cleaned_messages = []
#         for message in self.commit_messages:
#             cleaned_message = self._clean_message(message)
#             if cleaned_message:
#                 self.cleaned_messages.append(cleaned_message)
#         return self.cleaned_messages
    
#     def dynamic_filter(self):
#         filtered_data = self.clean()
#         # 第二步：统计分析
#         word_counter, feature_names, tfidf_matrix = self.statistical_analysis(filtered_data)
#         print("\n词频统计：")
#         for word, count in word_counter.most_common(10):
#             print(f"{word}: {count}")
        
#         # 第三步：动态词汇表构建
#         vocab, model = self.build_dynamic_lexicon(filtered_data)
#         print("\n动态词汇表：")
#         print(vocab)

#         # 第四步：动态规则更新
#         replacement_rules = self.update_rules(word_counter, vocab, model)
#         print("\n动态替换规则：")
#         for wrong, correct in replacement_rules.items():
#             print(f"'{wrong}' -> '{correct}'")
        
#         # 第五步：聚类与噪音检测
#         normal_words, noise_words, word_vectors, cluster_labels = self.cluster_and_detect_noise(vocab, model)
        
#         print("\n检测到的噪音词：")
#         print(noise_words)
        
#         # 第六步：基于聚类生成替换规则
#         cluster_replacement_rules = self.generate_rules_from_noise(noise_words, normal_words, model)
#         print("\n动态替换规则（基于聚类）：")
#         for wrong, correct in cluster_replacement_rules.items():
#             print(f"'{wrong}' -> '{correct}'")
#         # 合并所有替换规则
#         all_replacement_rules = {**replacement_rules, **cluster_replacement_rules}
#         # 应用所有替换规则
#         final_cleaned_data = [self.apply_replacement(text, all_replacement_rules) for text in filtered_data]

#         # # 第七步：可视化噪音分布
#         # self.plot_noise_distribution(word_vectors, cluster_labels)
#         return final_cleaned_data

#     def _clean_message(self, message):
#         # Remove hash values (40 characters of hexadecimal)
#         message = re.sub(r'\b[0-9a-f]{7,40}\b', '', message)

#         # Remove URLs
#         message = re.sub(r'http\S+', '', message)

#         # Remove issue references like (#1234), (fix #1234), (ref #1234)
#         message = re.sub(r'\(#[0-9]+\)', '', message)
#         message = re.sub(r'fix #[0-9]+', '', message)
#         message = re.sub(r'ref #[0-9]+', '', message)

#         # Remove [ci skip] and any similar tags
#         message = re.sub(r'\[.*?ci skip.*?\]', '', message)

#         # Remove unnecessary characters or symbols
#         message = re.sub(r'[\*\[\]\r\n]', ' ', message)

#         # Collapse multiple spaces into one
#         message = re.sub(r'\s+', ' ', message).strip()

#         # Return the cleaned message if it's not empty
#         return message if message else None

#     # 应用替换规则
#     def apply_replacement(self, text, rules):
#         for wrong, correct in rules.items():
#             text = re.sub(rf'\b{re.escape(wrong)}\b', correct, text)
#         return text

#     def statistical_analysis(self, texts, threshold=5):
#         # 计算词频
#         word_counter = Counter()
#         for text in texts:
#             words = nltk.word_tokenize(text)
#             word_counter.update(words)
        
#         # 计算TF-IDF
#         vectorizer = TfidfVectorizer()
#         tfidf_matrix = vectorizer.fit_transform(texts)
#         feature_names = vectorizer.get_feature_names_out()
        
#         return word_counter, feature_names, tfidf_matrix

#     def build_dynamic_lexicon(self, texts, vector_size=100, window=5, min_count=1, workers=4):
#         # 分词
#         tokenized_texts = [nltk.word_tokenize(text) for text in texts]
        
#         # 训练FastText模型
#         model = FastText(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        
#         # 构建词汇表
#         vocab = set()
#         for text in texts:
#             words = nltk.word_tokenize(text)
#             vocab.update(words)
        
#         return vocab, model

#     def update_rules(self, word_counter, vocab, model, frequency_threshold=2):
#         replacement_rules = {}
#         for word, count in word_counter.items():
#             if count < frequency_threshold and word in vocab:
#                 # 查找最相似的正确词
#                 similar_words = model.wv.most_similar(word, topn=1)
#                 if similar_words:
#                     similar_word, similarity = similar_words[0]
#                     if similarity > 0.7:  # 相似度阈值
#                         replacement_rules[word] = similar_word
#         return replacement_rules
    
#     def cluster_and_detect_noise(self, vocab, model, min_cluster_size=2, min_samples=1):
#         # 获取词向量
#         word_vectors = []
#         words = []
#         for word in vocab:
#             if word in model.wv:
#                 word_vectors.append(model.wv[word])
#                 words.append(word)
        
#         word_vectors = np.array(word_vectors)
        
#         # 使用HDBSCAN进行聚类
#         clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
#         cluster_labels = clusterer.fit_predict(word_vectors)
        
#         # 分离正常群组和噪音群组
#         normal_words = []
#         noise_words = []
#         for word, label in zip(words, cluster_labels):
#             if label == -1:
#                 noise_words.append(word)
#             else:
#                 normal_words.append(word)
        
#         return normal_words, noise_words, word_vectors, cluster_labels
    
#     def generate_rules_from_noise(self, noise_words, normal_words, model, similarity_threshold=0.7):
#         replacement_rules = {}
#         for noise_word in noise_words:
#             # 查找最相似的正常词
#             similar_words = model.wv.most_similar(noise_word, topn=5)
#             for similar_word, similarity in similar_words:
#                 if similar_word in normal_words and similarity >= similarity_threshold:
#                     replacement_rules[noise_word] = similar_word
#                     break
#         return replacement_rules

#     def plot_noise_distribution(self, word_vectors, cluster_labels):
#         # 使用PCA降维
#         pca = PCA(n_components=2)
#         reduced_vectors = pca.fit_transform(word_vectors)
        
#         plt.figure(figsize=(12, 8))
#         sns.scatterplot(x=reduced_vectors[:,0], y=reduced_vectors[:,1], hue=cluster_labels, palette='Set1', legend='full')
#         plt.title('Noise Distribution Clustering')
#         plt.xlabel('PCA Component 1')
#         plt.ylabel('PCA Component 2')
#         plt.legend(title='Cluster')
#         plt.show()




# def process_jsonl(file_path):
#     commit_messages = []
#     release_notes = []
#     cleaned_data = []
    
#     # 读取JSONL文件
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             # 提取需要的字段
#             commits = data.get('commit_messages', [])
#             release_note = data.get('release_note', {})
            
#             # 收集所有commit_messages
#             for commit in commits:
#                 # filtered_commit = rule_filtering(commit)
#                 # commit_messages.append(filtered_commit)
#                 commit_messages.append(commit)
            
#             # 收集所有release_note内容
#             for key, notes in release_note.items():
#                 for note in notes:
#                     # filtered_note = rule_filtering(note)
#                     # release_notes.append(filtered_note)
#                     release_notes.append(note)

#     return commit_messages, release_notes

# if __name__ == "__main__":
    
#     input_file = 'test.jsonl'
#     output_file = 'cleaned_test.jsonl'
#     print("读取和处理JSONL文件...")
#     commit_messages, release_notes = process_jsonl(input_file)
#     all_texts = commit_messages + release_notes
#     print(f"总共有 {len(all_texts)} 条文本进行处理。")


#     cleaner = ReleaseNoteCleaner(all_texts)
#     cleaned_messages = cleaner.dynamic_filter()

#     for message in cleaned_messages:
#         print(message)
