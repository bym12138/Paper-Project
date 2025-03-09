import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import itertools

class G(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device      
         
    def create(self, num_node):
        u1 = torch.arange(num_node, dtype=torch.int32)
        v1 = torch.arange(num_node, dtype=torch.int32)
        all_pairs = list(itertools.permutations(range(num_node), 2)) 
        u2 = [i[0] for i in all_pairs]
        v2 = [i[1] for i in all_pairs]
        u2 = torch.tensor(u2)
        v2 = torch.tensor(v2)
        graph_data = {
            ('sent', 'has', 'type'): (u1, v1)
        }
        g = dgl.heterograph(graph_data)
        g = g.to(self.device)
        return g

    def set_feature(self, g, sent_hidden, type_hidden):
        g.nodes['sent'].data['featu'] = sent_hidden
        g.nodes['type'].data['featu'] = type_hidden
        return g

    def forward(self, sent_hidden_batch, type_hidden_batch):
        g_batch = []
        for sent_hidden, type_hidden in zip(sent_hidden_batch, type_hidden_batch):
            num_node = 32
            g = self.create(num_node)
            g = self.set_feature(g, sent_hidden, type_hidden)
            g_batch.append(g)

        g = dgl.batch(g_batch)
        return g

























# # import pandas as pd
# # import torch
# # import dgl
# # from transformers import RobertaModel, RobertaTokenizer
# # import torch.nn as nn

# # # 加载数据
# # file_path = 'mult_class/data/class.csv'
# # data = pd.read_csv(file_path)

# # # 初始化RoBERTa的分词器和模型
# # tokenizer = RobertaTokenizer.from_pretrained('mult_class/roberta-base')
# # model = RobertaModel.from_pretrained('mult_class/roberta-base')

# # # 创建标签到索引的映射
# # all_labels = set()
# # for label_str in data['Categories']:
# #     labels = label_str.split(',')
# #     for label in labels:
# #         all_labels.add(label.strip())
# # label_to_index = {label: idx for idx, label in enumerate(all_labels)}

# # # 定义标签嵌入
# # num_labels = 9
# # label_embedding_dim = 16
# # label_embedding = nn.Embedding(num_labels, label_embedding_dim)  # 假设标签嵌入维度为16

# # # 辅助函数：将标签转换为索引
# # def process_labels(label_str, label_to_index):
# #     labels = label_str.split(',')
# #     label_indices = [label_to_index[label.strip()] for label in labels if label.strip() in label_to_index]
# #     return label_indices

# # # 创建标签到索引的映射
# # all_labels = set()
# # for label_str in data['Categories']:
# #     labels = label_str.split(',')
# #     for label in labels:
# #         all_labels.add(label.strip())
# # label_to_index = {label: idx for idx, label in enumerate(all_labels)}

# # # 将数据拆分成批次
# # batch_size = 32
# # num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)

# # # 处理每个批次
# # batches = []
# # for i in range(num_batches):
# #     batch_data = data.iloc[i*batch_size:(i+1)*batch_size]
# #     sentences = batch_data['Sentence'].tolist()
# #     labels = batch_data['Categories'].tolist()
    
# #     # 分词
# #     encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
# #     # 获取句子嵌入
# #     with torch.no_grad():
# #         outputs = model(**encoded_inputs)
# #     sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
    
# #     # 获取标签嵌入
# #     batch_label_indices = [process_labels(label_str, label_to_index) for label_str in labels]
    
# #     # 使用标签索引获取标签嵌入
# #     unique_label_indices = list(set([idx for sublist in batch_label_indices for idx in sublist]))
# #     unique_label_embeddings = label_embedding(torch.tensor(unique_label_indices))

# #     # 准备边的列表
# #     sentence_to_label_src = []
# #     sentence_to_label_dst = []
# #     label_to_sentence_src = []
# #     label_to_sentence_dst = []

# #     for idx, label_indices in enumerate(batch_label_indices):
# #         for label_idx in label_indices:
# #             label_node_idx = unique_label_indices.index(label_idx)
# #             sentence_to_label_src.append(idx)
# #             sentence_to_label_dst.append(label_node_idx)
# #             label_to_sentence_src.append(label_node_idx)
# #             label_to_sentence_dst.append(idx)

# #     # 创建异构图
# #     g = dgl.heterograph({
# #         ('sentence', 'contains', 'label'): (torch.tensor(sentence_to_label_src), torch.tensor(sentence_to_label_dst)),
# #         ('label', 'contained_by', 'sentence'): (torch.tensor(label_to_sentence_src), torch.tensor(label_to_sentence_dst))
# #     }, num_nodes_dict={'sentence': len(sentences), 'label': len(unique_label_indices)})
    
# #     # 存储批次数据
# #     batches.append((g, sentence_embeddings, unique_label_embeddings))

# # # 示例：显示第一个批次
# # print(batches[0])