import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

hyperparams = {
    "train_path": "noisyCleaner/data/train_type_score.jsonl",
    "test_path":  "noisyCleaner/data/test_type_score.jsonl",
    "save_dir":   "noisyCleaner/other_model/model/main",
    # 训练参数
    "num_epochs":   1,
    "batch_size":   8,
    "learning_rate":1e-3,

    # 模型结构
    "embed_dim":  128,  
    "hidden_dim": 128, 
    "num_classes":3,   

    # PPMI图
    "window_size":   3,
    "ppmi_threshold":0.1,

    # 依存图
    "corenlp_path": "noisyCleaner/data/stanford-corenlp-4.5.8",
    "alpha_dict": {
        "nsubj":   1.5,
        "dobj":    1.2,
        "amod":    0.8,
        "compound":1.0,
        "advmod":  1.3,
        "root":    1.0
    }
}


def build_ppmi_for_text(text: str, window_size=3, threshold=0.0):
    tokens = text.split()
    if not tokens:
        tokens = ["<EMPTY>"]
    vocab = sorted(list(set(tokens)))
    vocab2id = {w: i for i, w in enumerate(vocab)}
    V = len(vocab2id)

    cooccur_mat = np.zeros((V, V), dtype=np.float32)
    word_counts = np.zeros(V, dtype=np.float32)
    total_count = 0

    # 统计词频 & 共现
    for i, w in enumerate(tokens):
        wid = vocab2id[w]
        word_counts[wid] += 1
        total_count += 1
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        for j in range(start, end):
            if j == i:
                continue
            wj = tokens[j]
            wj_id = vocab2id[wj]
            cooccur_mat[wid, wj_id] += 1

    # 计算 PPMI
    adj_ppmi = np.zeros((V, V), dtype=np.float32)
    p = word_counts / (total_count + 1e-8)
    for i in range(V):
        for j in range(V):
            if cooccur_mat[i, j] > 0:
                p_ij = cooccur_mat[i, j] / (total_count + 1e-8)
                p_i = p[i]
                p_j = p[j]
                pmi = math.log2(p_ij / (p_i * p_j) + 1e-8)
                ppmi = max(pmi, 0.0)
                if ppmi > threshold:
                    adj_ppmi[i, j] = ppmi

    return adj_ppmi, vocab2id


def build_syntax_for_text(text: str, vocab2id: Dict[str,int], corenlp_path: str, alpha_dict: dict):
    from stanfordcorenlp import StanfordCoreNLP
    import math
    import numpy as np
    from collections import Counter

    tokens = text.split()
    if not tokens:
        tokens = ["<EMPTY>"]
    V = len(vocab2id)

    word_freq = Counter(tokens)
    sum_words = len(tokens)
    def tf_idf(w):
        tf = word_freq[w] / (sum_words + 1e-8)
        return tf

    adjacency_syntax = np.zeros((V, V), dtype=np.float32)
   
    nlp = StanfordCoreNLP(corenlp_path, lang='en') 
    parse_result = nlp.dependency_parse(" ".join(tokens))
    for (rel, gov, dep) in parse_result:
        if gov<=0 or dep<=0 or gov>len(tokens) or dep>len(tokens):
            continue
        gw = tokens[gov-1]
        dw = tokens[dep-1]
        if gw not in vocab2id or dw not in vocab2id:
            continue
        gw_id = vocab2id[gw]
        dw_id = vocab2id[dw]

        alpha_r = alpha_dict.get(rel, 1.0)
        w_edge = alpha_r * (tf_idf(gw) + tf_idf(dw)) / 2.0
        adjacency_syntax[gw_id, dw_id] += w_edge
        adjacency_syntax[dw_id, gw_id] += w_edge  # 无向
    
    nlp.close()
    return adjacency_syntax


def parse_jsonl_and_build_graphs(filepath: str) -> List[Dict]:

    from math import log2
    from tqdm import tqdm 

    alpha_dict = hyperparams["alpha_dict"]
    corenlp_path = hyperparams["corenlp_path"]
    window_size  = hyperparams["window_size"]
    ppmi_thr     = hyperparams["ppmi_threshold"]

    data_list = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Reading {len(lines)} lines from {filepath} ...")

    for line in tqdm(lines):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        commit_msgs = obj.get("commit_messages", [])
        for cm in commit_msgs:
            msg_text = cm.get("message", "").strip()
            msg_score= cm.get("score", 0)
            if msg_score not in [1,2,3]:
                continue
            adj_ppmi, vocab2id = build_ppmi_for_text(
                msg_text,
                window_size=window_size,
                threshold=ppmi_thr
            )
            adj_syntax = build_syntax_for_text(
                msg_text,
                vocab2id,
                corenlp_path=corenlp_path,
                alpha_dict=alpha_dict
            )
            label = msg_score - 1
            data_list.append({
                "adj_ppmi": adj_ppmi,
                "adj_syntax": adj_syntax,
                "vocab2id": vocab2id,
                "label": label
            })
    return data_list


class GraphScoreDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        return item

def collate_fn_graph(batch):
    return batch 

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(SimpleGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation
    
    def forward(self, x, A):
        # x: [V, in_dim], A: [V, V]
        x = A @ x @ self.weight  # => [V, out_dim]
        if self.activation:
            x = self.activation(x)
        return x

class MultiBranchGNN(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128, num_classes=3):
        super(MultiBranchGNN, self).__init__()
        self.gcn_ppmi = SimpleGCNLayer(in_dim=128, out_dim=hidden_dim, activation=nn.ReLU())
        self.gcn_syntax = SimpleGCNLayer(in_dim=128, out_dim=hidden_dim, activation=nn.ReLU())
        self.fuse = nn.Linear(hidden_dim*2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, X_ppmi, A_ppmi, X_syntax, A_syntax):
        out1 = self.gcn_ppmi(X_ppmi, A_ppmi)       # => [V1, hidden_dim]
        out2 = self.gcn_syntax(X_syntax, A_syntax) # => [V2, hidden_dim]
        fuse1 = out1.mean(dim=0)  # [hidden_dim]
        fuse2 = out2.mean(dim=0)  # [hidden_dim]

        fusion = torch.cat([fuse1, fuse2], dim=-1)  # [hidden_dim*2]
        fusion = self.fuse(fusion)                  # [hidden_dim]
        logits = self.classifier(fusion.unsqueeze(0))  # [1, num_classes]
        return logits


def train_and_evaluate():
    num_epochs   = hyperparams["num_epochs"]
    batch_size   = hyperparams["batch_size"]
    lr           = hyperparams["learning_rate"]
    embed_dim    = hyperparams["embed_dim"]
    hidden_dim   = hyperparams["hidden_dim"]
    num_classes  = hyperparams["num_classes"]
    train_path   = hyperparams["train_path"]
    test_path    = hyperparams["test_path"]
    save_dir     = hyperparams["save_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Building training graphs...")
    train_data_list = parse_jsonl_and_build_graphs(train_path)
    print(f"Got {len(train_data_list)} training samples.")
    train_dataset = GraphScoreDataset(train_data_list)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_graph
    )

    print("Building testing graphs...")
    test_data_list = parse_jsonl_and_build_graphs(test_path)
    print(f"Got {len(test_data_list)} testing samples.")
    test_dataset = GraphScoreDataset(test_data_list)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_graph
    )

    model = MultiBranchGNN(embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * num_epochs
    global_step = 0

    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            item = batch[0]
            A_ppmi_np = item["adj_ppmi"]              # shape=[V1,V1]
            vocab2id  = item["vocab2id"]
            V1 = A_ppmi_np.shape[0]
            I_ppmi = np.eye(V1, dtype=np.float32)
            A_syntax_np = item["adj_syntax"]          # shape=[V2,V2]
            V2 = A_syntax_np.shape[0]
            I_syntax = np.eye(V2, dtype=np.float32)
            label = item["label"]  # 0/1/2
            label_t = torch.tensor([label], dtype=torch.long, device=device)

            # 转为 torch
            A_ppmi_t   = torch.tensor(A_ppmi_np,   dtype=torch.float, device=device)
            A_syntax_t = torch.tensor(A_syntax_np, dtype=torch.float, device=device)
            X_ppmi_t   = torch.tensor(I_ppmi,      dtype=torch.float, device=device)
            X_syntax_t = torch.tensor(I_syntax,    dtype=torch.float, device=device)

            logits = model(X_ppmi_t, A_ppmi_t, X_syntax_t, A_syntax_t)
            loss = criterion(logits, label_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 10 == 0:
                print(f"[step/total: {global_step}/{total_steps}, loss: {loss.item():.4f}]")

    model.eval()
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    all_preds = []
    all_labels= []

    with torch.no_grad():
        for batch in test_loader:
            item = batch[0]
            A_ppmi_np   = item["adj_ppmi"]
            A_syntax_np = item["adj_syntax"]
            V1 = A_ppmi_np.shape[0]
            V2 = A_syntax_np.shape[0]
            I_ppmi = np.eye(V1, dtype=np.float32)
            I_syntax = np.eye(V2, dtype=np.float32)
            label = item["label"]

            A_ppmi_t   = torch.tensor(A_ppmi_np,   dtype=torch.float, device=device)
            A_syntax_t = torch.tensor(A_syntax_np, dtype=torch.float, device=device)
            X_ppmi_t   = torch.tensor(I_ppmi,      dtype=torch.float, device=device)
            X_syntax_t = torch.tensor(I_syntax,    dtype=torch.float, device=device)

            logits = model(X_ppmi_t, A_ppmi_t, X_syntax_t, A_syntax_t)  # [1,3]
            pred = torch.argmax(logits, dim=-1).item()
            
            all_preds.append(pred)
            all_labels.append(label)

    acc = accuracy_score(all_labels, all_preds)
    prec_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec_macro  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted= f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("===== Evaluation Results =====")
    print(f"Accuracy:        {acc:.4f}")
    print(f"PrecisionMacro:  {prec_macro:.4f}")
    print(f"RecallMacro:     {rec_macro:.4f}")
    print(f"F1Macro:         {f1_macro:.4f}")
    print(f"F1Weighted:      {f1_weighted:.4f}")


    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "eval_results.txt"), "w", encoding="utf-8") as f:
        f.write("===== Evaluation Results =====\n")
        f.write(f"Accuracy:        {acc:.4f}\n")
        f.write(f"PrecisionMacro:  {prec_macro:.4f}\n")
        f.write(f"RecallMacro:     {rec_macro:.4f}\n")
        f.write(f"F1Macro:         {f1_macro:.4f}\n")
        f.write(f"F1Weighted:      {f1_weighted:.4f}\n")

if __name__ == "__main__":
    train_and_evaluate()
