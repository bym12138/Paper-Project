# 标签的embedding
import torch
import torch.nn as nn

labels = ["New Features", "Bug Fixes", "Improvements", "Update", "Security", 
          "Changes", "Deprecations and Removals", "Documentation and Tooling", "Miscellaneous"]

label_to_id = {label: idx for idx, label in enumerate(labels)}



embedding_dim = 768 
label_embeddings = nn.Embedding(num_embeddings=len(labels), embedding_dim=embedding_dim)

def get_label_embedding(label):
    label_id = label_to_id[label]
    label_id = [0,2,4,2,5,6,8,4,2,2,5,2,6,6,3,8,6,4]
    label_id_tensor = torch.tensor(label_id, dtype=torch.long)
    label_embed = label_embeddings(label_id_tensor)
    a = label_embed.shape
    return label_embed

label = "New Features"
label_embedding = get_label_embedding(label)

print(f"Label: {label}")
print(f"Embedding: {label_embedding}")
