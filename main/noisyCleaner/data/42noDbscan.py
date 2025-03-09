import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from transformers import RobertaModel, RobertaTokenizer

model_name = "noisyCleaner/roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
model.eval() 
with open('noisyCleaner/data/test_ori.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
sampled_lines = random.sample(lines, min(12000, len(lines)))

data_words = []
for line in sampled_lines:
    obj = json.loads(line)
    commit_messages = obj.get('commit_messages', [])
    for msg in commit_messages:
        words = msg.split()  
        data_words.extend(words)
words = list(set(data_words))

embeddings = []
valid_words = []
for w in words:
    token = w.lower().strip()
    if token == "":
        continue
    inputs = tokenizer(token, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    if outputs.last_hidden_state.shape[1] > 2:
        word_embedding = outputs.last_hidden_state[0][1:-1].mean(dim=0)
    else:
        word_embedding = outputs.last_hidden_state[0].mean(dim=0)
    embeddings.append(word_embedding.cpu().numpy())
    valid_words.append(w)
embeddings = np.array(embeddings)
print(f"共获得 {len(valid_words)} 个词的向量。")

dbscan = DBSCAN(eps=0.6, min_samples=2) # 0.5
labels = dbscan.fit_predict(embeddings)
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

normal_indices = np.where(labels != -1)[0]
noise_indices = np.where(labels == -1)[0]
isolated_noise_indices = []
threshold = 0.1 
for idx in noise_indices:
    point = embeddings_2d[idx]
    noise_points = embeddings_2d[noise_indices]
    dists = np.linalg.norm(noise_points - point, axis=1)
    if np.sum(dists < threshold) <= 1:
        isolated_noise_indices.append(idx)
isolated_noise_indices = np.array(isolated_noise_indices)

plt.rcParams["font.family"] = "SimHei"          
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(embeddings_2d[normal_indices, 0], embeddings_2d[normal_indices, 1],
                c='blue', marker='o', s=100, label='正常词汇')
if len(isolated_noise_indices) > 0:
    axes[0].scatter(embeddings_2d[isolated_noise_indices, 0], embeddings_2d[isolated_noise_indices, 1],
                    c='red', marker='x', s=100, label='离群噪音点')
axes[0].set_title("使用 DBSCAN 聚类的词向量分布")
axes[0].set_xlabel("主成分 1（PCA 降维）")
axes[0].set_ylabel("主成分 2（PCA 降维）")
axes[0].legend()

# 右图：未经 DBSCAN 聚类，仅展示所有词向量（不标记噪音）
axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                c='blue', marker='o', s=100, label='词向量')
axes[1].set_title("未经 DBSCAN 聚类的词向量分布")
axes[1].set_xlabel("主成分 1（PCA 降维）")
axes[1].set_ylabel("主成分 2（PCA 降维）")
axes[1].legend()

plt.suptitle("软件发行说明文本词向量对比图", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
