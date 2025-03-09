# 标签的矩阵化和标签的向量化结合
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sklearn.decomposition import PCA
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
labels = ["New Features", "Bug Fixes", "Improvements", "Update", "Security", 
          "Changes", "Deprecations and Removals", "Documentation and Tooling", "Miscellaneous"]

label_to_id = {label: idx for idx, label in enumerate(labels)}

embedding_dim = 768
label_embeddings = nn.Embedding(num_embeddings=len(labels), embedding_dim=embedding_dim)

def get_label_embedding(label):
    label_id = label_to_id[label]
    label_id_tensor = torch.tensor(label_id, dtype=torch.long)
    label_embed = label_embeddings(label_id_tensor)
    return label_embed

documents = []
file_path = 'mult_class/data/class.csv'
df = pd.read_csv(file_path)
for index, row in df.iterrows():
    entry = {
        "text": row['Sentence'],
        "labels": [label.strip() for label in row['Categories'].split(',')]
    }
    documents.append(entry)

labels = sorted(list(set([label for doc in documents for label in doc["labels"]])))
label_to_index = {label: i for i, label in enumerate(labels)}

X = np.zeros((len(labels), len(labels)))

for doc in documents:
    doc_labels = doc["labels"]
    for i in range(len(doc_labels)):
        for j in range(i + 1, len(doc_labels)):
            idx_i = label_to_index[doc_labels[i]]
            idx_j = label_to_index[doc_labels[j]]
            X[idx_i][idx_j] += 1
            X[idx_j][idx_i] += 1 

A = X.copy()
np.fill_diagonal(A, 1)
D = np.diag(np.sum(A, axis=1))

D_inv_sqrt = np.linalg.inv(np.sqrt(D))
A_tilde = D_inv_sqrt @ A @ D_inv_sqrt

print("归一化邻接矩阵 A_tilde:\n", A_tilde)
aa=A_tilde.shape
embedding_matrix = np.zeros((len(labels), embedding_dim))
a=embedding_matrix.shape
for label in labels:
    embedding = get_label_embedding(label).detach().numpy()
    embedding_matrix[label_to_index[label]] = embedding

linear = nn.Linear(9, embedding_dim)
A_tilde_transformed = linear(torch.tensor(A_tilde, dtype=torch.float32))  # shape: (9, 768)

embedding_matrix_with_batch = torch.tensor(embedding_matrix, dtype=torch.float32).unsqueeze(1)  # shape: (9, 1, 768)
A_tilde_transformed_with_batch = A_tilde_transformed.unsqueeze(1)  # shape: (9, 1, 768)

attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8)

combined_matrix, _ = attention(embedding_matrix_with_batch, embedding_matrix_with_batch, A_tilde_transformed_with_batch)

combined_matrix = combined_matrix.squeeze(1)  # shape: (9, 768)

print("Combined Matrix Shape:", combined_matrix.shape)
