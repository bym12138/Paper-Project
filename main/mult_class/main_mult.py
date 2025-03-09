import os
import json
import jsonlines
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizer, RobertaForMaskedLM
import dgl
import dgl.nn as dglnn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class CommitDataset(Dataset):
    def __init__(self, 
                 jsonl_path, 
                 roberta_model_path, 
                 label_mapping,
                 sim_threshold=0.5, 
                 max_length=64, 
                 device="cpu"):
        super().__init__()
        self.samples = []
        self.label_mapping = label_mapping
        self.sim_threshold = sim_threshold
        self.max_length = max_length
        self.device = device
        self.bare_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path)
        self.bare_model = RobertaForMaskedLM.from_pretrained(roberta_model_path).to(device)
        self.bare_model.eval()
        for param in self.bare_model.parameters():
            param.requires_grad = False

        with jsonlines.open(jsonl_path, "r") as reader:
            for obj in reader:
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def build_graph_for_jsonl(self, sample):
        commit_messages = sample.get("commit_messages", [])
        sentence_texts = []
        sentence_labels = []
        for msg in commit_messages:
            text = msg["message"].strip()
            t = msg.get("type", "").strip().lower()
            label_id = self.label_mapping.get(t, -1)
            sentence_texts.append(text)
            sentence_labels.append(label_id)

        num_sentence_nodes = len(sentence_texts)
        num_label_nodes = len(self.label_mapping)  # 9
        total_nodes = num_label_nodes + num_sentence_nodes

        edges_src = []
        edges_dst = []
        for i, lb in enumerate(sentence_labels):
            if lb != -1:
                s_id = num_label_nodes + i
                c_id = lb
                edges_src.append(s_id)
                edges_dst.append(c_id)
                edges_src.append(c_id)
                edges_dst.append(s_id)

        with torch.no_grad():
            cls_embs = []
            for text in sentence_texts:
                inputs = self.bare_tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=self.max_length
                ).to(self.device)

                outputs = self.bare_model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
                cls_vector = last_hidden[0, 0, :].clone().detach()  # 取 <s> 位置
                cls_embs.append(cls_vector)

            cls_embs = torch.stack(cls_embs, dim=0)  # [num_sentence_nodes, hidden_dim]

            for i in range(num_sentence_nodes):
                for j in range(i+1, num_sentence_nodes):
                    emb_i = cls_embs[i]
                    emb_j = cls_embs[j]
                    cos_sim = F.cosine_similarity(
                        emb_i.unsqueeze(0), emb_j.unsqueeze(0)
                    ).item()
                    if cos_sim > self.sim_threshold:
                        si = num_label_nodes + i
                        sj = num_label_nodes + j
                        edges_src.append(si)
                        edges_dst.append(sj)
                        edges_src.append(sj)
                        edges_dst.append(si)

        g = dgl.graph(
            (edges_src, edges_dst),
            num_nodes=total_nodes
        )

        node_feats = torch.zeros(total_nodes, 128)
        g.ndata["feat"] = node_feats

        node_labels = [-1]*num_label_nodes + sentence_labels

        return g, sentence_texts, node_labels

def collate_fn(samples, dataset_obj):
    graphs = []
    all_sentence_texts = []
    all_node_labels = []
    for sample in samples:
        g, sentence_texts, node_labels = dataset_obj.build_graph_for_jsonl(sample)
        graphs.append(g)
        all_sentence_texts.append(sentence_texts)
        all_node_labels.append(node_labels)

    batched_graph = dgl.batch(graphs)
    return batched_graph, all_sentence_texts, all_node_labels

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, num_layers=2, aggregator_type='mean'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, hidden_size, aggregator_type))
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, aggregator_type))
        if num_layers > 1:
            self.layers.append(dglnn.SAGEConv(hidden_size, out_feats, aggregator_type))
        self.num_layers = num_layers

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < self.num_layers - 1:
                h = F.relu(h)
        return h

class MultiTemplateBayesianModel(nn.Module):
    def __init__(self,
                 roberta_model_path,       
                 label_list,                
                 template_strings,          # list[str] (含 <mask>)
                 graph_in_feats=128,
                 graph_hidden=128,
                 graph_out=128,
                 num_layers=2,
                 lambda_w=1e-4,            # 正则项
                 device="cpu"):
        super().__init__()


        self.graphsage = GraphSAGE(in_feats=graph_in_feats,
                                   hidden_size=graph_hidden,
                                   out_feats=graph_out,
                                   num_layers=num_layers)

        self.template_strings = template_strings
        self.num_templates = len(template_strings)
        self.template_weight = nn.Parameter(torch.zeros(self.num_templates, graph_out))
        nn.init.xavier_normal_(self.template_weight)

        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path)
        self.mlm_model = RobertaForMaskedLM.from_pretrained(roberta_model_path)
        for param in self.mlm_model.roberta.parameters():
            param.requires_grad = False
        self.mlm_model.train()  

        self.label_list = label_list
        self.num_classes = len(label_list)
        label_ids = []
        for lbl_word in label_list:
            ids = self.tokenizer.encode(lbl_word, add_special_tokens=False)
            label_ids.append(ids[0]) 
        self.label_ids = label_ids

        self.lambda_w = lambda_w
        self.device = device
        self.to(device)

    def forward(self, batched_graph, all_sentence_texts, all_node_labels=None):
        node_feats = batched_graph.ndata["feat"].to(self.device)
        h_graph_all = self.graphsage(batched_graph, node_feats)  # [N, graph_out]

        # unbatch
        gs = dgl.unbatch(batched_graph)
        offset = 0

        all_sentence_info = []
        all_sentence_h    = []

        for i, small_g in enumerate(gs):
            num_nodes_small = small_g.num_nodes()
            local_h_graph = h_graph_all[offset : offset + num_nodes_small]
            offset += num_nodes_small

            sentence_text_list = all_sentence_texts[i]
            sentence_count = len(sentence_text_list)
            sentence_h_graph = local_h_graph[9 : 9 + sentence_count]

            for s_idx in range(sentence_count):
                all_sentence_info.append( (i, s_idx) )
                all_sentence_h.append(sentence_h_graph[s_idx])
        all_sentence_h = torch.stack(all_sentence_h, dim=0) if len(all_sentence_h)>0 else torch.empty((0,0), device=self.device)
        S = len(all_sentence_info)

        if S == 0:
            if all_node_labels is not None:
                return torch.tensor(0.0, device=self.device), {}
            else:
                return []

        prompt_texts = []
        mapping_index = []

        for s_idx in range(S):
            g_i, local_s_i = all_sentence_info[s_idx]
            text_str = all_sentence_texts[g_i][local_s_i]

            for t_i in range(self.num_templates):
                prompt_str = self.template_strings[t_i] + " " + text_str
                prompt_texts.append(prompt_str)
                mapping_index.append( (s_idx, t_i) )

        encodings = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.mlm_model(
            encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            output_hidden_states=False,
            return_dict=True
        )
        logits = outputs.logits  # shape=[S*T, seq_len, vocab_size]

        mask_token_id = self.tokenizer.mask_token_id
        input_ids_2d = encodings["input_ids"]  # [S*T, seq_len]

        mask_indices = [-1]*(S*self.num_templates)

        row_col = (input_ids_2d == mask_token_id).nonzero(as_tuple=False)
        for rc in row_col:
            r, c = rc.tolist()
            if mask_indices[r] == -1: 
                mask_indices[r] = c
        p_y_stacked = []
        for r in range(S*self.num_templates):
            if mask_indices[r] < 0:
                p_each_class = torch.ones(self.num_classes, device=self.device)/(self.num_classes)
                p_y_stacked.append(p_each_class)
            else:
                c = mask_indices[r]
                row_logits = logits[r, c, :]  # [vocab_size]
                label_logits = row_logits[self.label_ids]  # [num_classes]
                p_each_class = F.softmax(label_logits, dim=-1)  # [num_classes]
                p_y_stacked.append(p_each_class)

        p_y_stacked = torch.stack(p_y_stacked, dim=0)  # [S*T, num_classes]

        # reshape => [S, T, num_classes]
        p_y_stacked = p_y_stacked.view(S, self.num_templates, self.num_classes)

        all_losses = []
        all_preds  = []
        all_trues  = []

        alpha_raw = torch.matmul(all_sentence_h, self.template_weight.transpose(0,1))  # [S, T]
        alpha = F.softmax(alpha_raw, dim=-1)  # [S, T]

        if all_node_labels is not None:
            big_true_labels = []
            for (g_idx, local_s_idx) in all_sentence_info:
                node_labels = all_node_labels[g_idx]
                sentence_true_labels = node_labels[9:]  # list of int
                if local_s_idx < len(sentence_true_labels):
                    lab = sentence_true_labels[local_s_idx]
                else:
                    lab = -1
                big_true_labels.append(lab)

            for s in range(S):
                true_label = big_true_labels[s]
                if true_label is not None and true_label != -1:
                    p_label_i = p_y_stacked[s, :, true_label]
                    log_probs = torch.log(p_label_i + 1e-12)  # [T]
                    loss_node = - torch.sum(alpha[s] * log_probs)
                    all_losses.append(loss_node)
                    p_y_fused = torch.sum(alpha[s].unsqueeze(-1) * p_y_stacked[s], dim=0)
                    y_pred = torch.argmax(p_y_fused).item()
                    all_preds.append(y_pred)
                    all_trues.append(true_label)
                else:
                    p_y_fused = torch.sum(alpha[s].unsqueeze(-1) * p_y_stacked[s], dim=0)
                    y_pred = torch.argmax(p_y_fused).item()
                    all_preds.append(y_pred)
                    all_trues.append(-1)

            if len(all_losses) > 0:
                loss_batch = torch.mean(torch.stack(all_losses))
            else:
                loss_batch = torch.tensor(0.0, device=self.device)

            reg_w = torch.sum(self.template_weight * self.template_weight)
            loss_batch = loss_batch + self.lambda_w * reg_w

            final_preds = []
            final_trues = []
            for p, t in zip(all_preds, all_trues):
                if t != -1:
                    final_preds.append(p)
                    final_trues.append(t)

            if len(final_trues) == 0:
                metrics = {}
            else:
                acc = accuracy_score(final_trues, final_preds)
                pre = precision_score(final_trues, final_preds, average="macro", zero_division=0)
                rec = recall_score(final_trues, final_preds, average="macro", zero_division=0)
                f1 = f1_score(final_trues, final_preds, average="macro", zero_division=0)
                metrics = {
                    "acc": acc,
                    "precision": pre,
                    "recall": rec,
                    "f1": f1
                }
            return loss_batch, metrics
        else:
            all_preds = []
            for s in range(S):
                p_y_fused = torch.sum(alpha[s].unsqueeze(-1) * p_y_stacked[s], dim=0)
                y_pred = torch.argmax(p_y_fused).item()
                all_preds.append(y_pred)
            return all_preds

def train_loop(model, dataloader, optimizer, device="cpu", num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        print(f"=== Epoch {epoch+1}/{num_epochs} ===")
        total_loss = 0.0
        count = 0

        for batch_idx, (batched_graph, all_sentence_texts, all_node_labels) in enumerate(
            tqdm(dataloader, desc="Train", ncols=100)
        ):
            batched_graph = batched_graph.to(device)

            optimizer.zero_grad()
            loss_batch, metrics = model(batched_graph, all_sentence_texts, all_node_labels)
            loss_batch.backward()
            optimizer.step()

            total_loss += loss_batch.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        print(f"Epoch[{epoch+1}], avg_loss={avg_loss:.4f}")


def evaluate(model, dataloader, device="cpu"):
    model.eval()
    all_preds = []
    all_trues = []
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (batched_graph, all_sentence_texts, all_node_labels) in enumerate(
            tqdm(dataloader, desc="Eval", ncols=100)
        ):
            batched_graph = batched_graph.to(device)
            loss_batch, metrics = model(batched_graph, all_sentence_texts, all_node_labels)
            total_loss += loss_batch.item()
            count += 1

            preds_in_batch = model(batched_graph, all_sentence_texts, None)
            for i, g in enumerate(dgl.unbatch(batched_graph)):
                node_labels = all_node_labels[i]
                sentence_labels = node_labels[9:]
                all_trues.extend(sentence_labels)
            all_preds.extend(preds_in_batch)

    final_preds = []
    final_trues = []
    idx_all = 0
    for t in all_trues:
        if t != -1:
            final_preds.append(all_preds[idx_all])
            final_trues.append(t)
        idx_all += 1

    avg_loss = total_loss / max(count, 1)
    acc = accuracy_score(final_trues, final_preds)
    pre = precision_score(final_trues, final_preds, average="macro", zero_division=0)
    rec = recall_score(final_trues, final_preds, average="macro", zero_division=0)
    f1 = f1_score(final_trues, final_preds, average="macro", zero_division=0)

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1
    }
    return metrics

def main():
    # A. 配置
    train_file = "mult_class/data/train_type.jsonl"
    test_file  = "mult_class/data/test_type.jsonl"
    roberta_path = "mult_class/roberta-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "mult_class//model/main"
    os.makedirs(save_dir, exist_ok=True)
    label_list = [
        "fixes",          
        "miscellaneous",  
        "feature",
        "remove",
        "security",
        "documentation",
        "changes",
        "improve",
        "update"
    ]
    label_mapping = {lbl: i for i, lbl in enumerate(label_list)}
    train_dataset_obj = CommitDataset(
        jsonl_path=train_file,
        roberta_model_path=roberta_path,
        label_mapping=label_mapping,
        sim_threshold=0.5,
        max_length=64,
        device=device
    )
    test_dataset_obj  = CommitDataset(
        jsonl_path=test_file,
        roberta_model_path=roberta_path,
        label_mapping=label_mapping,
        sim_threshold=0.5,
        max_length=64,
        device=device
    )

    def collate_wrapper_train(batch):
        return collate_fn(batch, train_dataset_obj)
    train_loader = DataLoader(
        train_dataset_obj,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_wrapper_train
    )

    def collate_wrapper_test(batch):
        return collate_fn(batch, test_dataset_obj)
    test_loader  = DataLoader(
        test_dataset_obj,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_wrapper_test
    )

    template_strings = [
        "this is type of <mask>",
        "this is <mask> type",
        "identified as <mask>",
        "classified as <mask>",
        "it belongs to the <mask> type"
    ]

    model = MultiTemplateBayesianModel(
        roberta_model_path=roberta_path,
        label_list=label_list,
        template_strings=template_strings,
        graph_in_feats=128,
        graph_hidden=128,
        graph_out=128,
        num_layers=2,
        lambda_w=1e-4,
        device=device
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_loop(model, train_loader, optimizer, device=device, num_epochs=3)
    metrics = evaluate(model, test_loader, device=device)
    print("==== Evaluation Results ====")
    for k,v in metrics.items():
        if isinstance(v, float):
            print(f"{k} = {v:.4f}")
        else:
            print(f"{k} = {v}")
    model_path = os.path.join(save_dir, "multi_template_mlm_model_frozen_batch.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
