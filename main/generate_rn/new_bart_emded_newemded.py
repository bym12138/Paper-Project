# embedding 改了
from transformers import BertTokenizer, BertModel, BartConfig, BartTokenizer, BartModel
import torch
from GATLayer import GATLayer
import torch.nn as nn

import dgl

from custom_modeling_bart import CustomBartForConditionalGeneration
from torch.utils.data import DataLoader
from torch.optim import AdamW

import json
from classifier import Classifier
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from transformers import Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import itertools
import torch.nn.functional as F

class BartEmbedding():

    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model.to(device)

    # 使用bart编码器
    def sent_hidden_state(self, doc):
        with torch.no_grad():  # 禁用梯度计算以减少内存使用
            input_ids_list = [
                self.tokenizer.encode(sentence,
                                      add_special_tokens=True,
                                      padding="max_length",
                                      max_length=256,
                                      truncation=True,
                                      return_tensors='pt') for sentence in doc
                if sentence
            ]
            embeddings = []
            for input_ids in input_ids_list:
                input_ids = input_ids.to(device)
                outputs = self.model(input_ids)
                embedding = outputs.last_hidden_state[:, 0, :]
                #使用bart-base
                linnet_sent = nn.Linear(768, 512, bias=False).to(device)

                embedding = linnet_sent(embedding)

                embeddings.append(embedding)

        # 将张量列表转换为张量
        embeddings = torch.cat(embeddings, dim=0)
        shape = embeddings.shape  # (len(doc), sequence_length, hidden_size)
        return embeddings

class G(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # num_g = len(sent_hidden_batch) # sent_hidden是个batch，算出一共num_g个图
        r'''
            sent_raw_batch：[doc1（有不一定数量的sent）,doc2,doc3,doc4...batch_size]
            sent_hidden_batch: [(len(doc), seq_len, hidden_dim),(),(),...,batch_size]
            type_hidden_batch: [(len(doc), hidden_dim),(),()...,batch_size]    
        '''
              
    def create(self, num_node):
        u1 = torch.arange(num_node, dtype=torch.int32)
        v1 = torch.arange(num_node, dtype=torch.int32)
        all_pairs = list(itertools.permutations(range(num_node), 2)) # 产生所有可能的两两组合
        u2 = [i[0] for i in all_pairs]
        v2 = [i[1] for i in all_pairs]
        u2 = torch.tensor(u2)
        v2 = torch.tensor(v2)
        graph_data = {
            ###('sent', 'interact', 'sent'): (u1, v1),
            ('sent', 'has', 'type'): (u1, v1)#(u2, v2)
        }
        # 可以试试边加权
        g = dgl.heterograph(graph_data)
        # g = dgl.to_bidirected(g)  #设置无向图出错
        # g.edges()
        g = g.to(self.device)
        return g

    def set_feature(self, g, sent_hidden, type_hidden):
        #设置特征向量
        shape = sent_hidden.shape  # (len(doc),512)
        shape1 = type_hidden.shape  # (len(doc),64)
        g.nodes['sent'].data['featu'] = sent_hidden
        g.nodes['type'].data['featu'] = type_hidden
        #暂时不用边的特征？ g.edata['he'] = torch.ones(edge_num, dtype=torch.float32).to(device)  # 暂时用1随机初始化每条边的权重
        #g = g.to('cuda')
        return g

    def forward(self, sent_raw_batch, sent_hidden_batch, type_hidden_batch):
        g_batch = []
        for sent_raw, sent_hidden, type_hidden in zip(sent_raw_batch, sent_hidden_batch, type_hidden_batch):
            num_node = len(sent_raw)
            g = self.create(num_node)
            g = self.set_feature(g, sent_hidden, type_hidden)
            g_batch.append(g)

        g = dgl.batch(g_batch)
        return g

class Graph(nn.Module): ## 加入继承nn.Module，用于opti
    def __init__(self, device):
        super().__init__()
        self.sent_node_feat_dim = 512  #1024#768  # 左侧节点特征维度
        self.type_node_feat_dim = 512  # 右侧节点特征维度
        self.edge_feat_dim = 1  # 边特征维度
        self.out_dim = 768  # 输出的维度 1024 bart-base
        self.common_dim = 1024
        self.device = device

        # 定义GAT层和线性层
        self.gat_layer = GATLayer(self.out_dim)
        self.linnet_sent = nn.Linear(self.sent_node_feat_dim, self.out_dim, bias=False)
        self.linnet_type = nn.Linear(self.type_node_feat_dim, self.out_dim, bias=False)
        

    def gat_nodes(self, graph):
        # 使用GAT层融合sent和type节点特征
        h_sent = self.linnet_sent(graph.nodes['sent'].data['featu'])
        h_type = self.linnet_type(graph.nodes['type'].data['featu'])   
        h_combined = self.gat_layer(graph, h_sent, h_type)
        h_combined = h_combined['sent']
        return h_combined


    def forward(self, graph):
        h_combined = self.gat_nodes(graph)
        return h_combined


class ExampleSet(Dataset):
    def __init__(self, jsonl, tokenizer):
        self.data = self.load_data(jsonl)
        self.tokenizer = tokenizer

    def load_data(self, jsonl):
        Data = {}
        i = 0
        for idx, item in enumerate(jsonl):
            # 循环其中一条messages，一个条组doc
            doc = item['single_commit_messages']
            summary = item['single_releases_note']
            Data[i] = {'doc': doc, 'summary': summary}
            i += 1
        return Data

    def collate_fn(self, batch_samples):
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            batch_inputs.append(sample['doc'])
            batch_targets.append(sample['summary'])

        input_ids, labels = self.encode(batch_inputs, batch_targets)
        sent_list = [batch_input.split(',') for batch_input in batch_inputs]
        batch_data = dict()
        batch_data['input_ids'] = input_ids['input_ids']
        batch_data['labels'] = labels['input_ids']
        batch_data['attention_mask'] = input_ids['attention_mask']
        batch_data['doc'] = sent_list
        # batch_data['summary'] = batch_targets
        return batch_data

    def encode(self, batch_inputs, batch_targets):
        input_ids = self.tokenizer(batch_inputs, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
        labels = self.tokenizer(batch_targets, padding="max_length", max_length=100, truncation=True, return_tensors="pt")
        return (input_ids, labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class AttentionEmbedding(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super(AttentionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        # 注意力层的权重
        self.attention_weights = nn.Parameter(torch.randn(embedding_dim, embedding_dim))

    def forward(self, x):
        # x: [num_classes]
        # 为了适应没有batch_size的情况，我们首先增加一个维度来模拟batch_size=1
        x = x.unsqueeze(0)  # [1, num_classes]
        embedded = self.embedding(x)  # [1, num_classes, embedding_dim]
        
        # 计算注意力分数
        attention_scores = torch.matmul(embedded, self.attention_weights)  # [1, num_classes, embedding_dim]
        attention_scores = torch.matmul(attention_scores, embedded.transpose(1, 2))  # [1, num_classes, num_classes]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [1, num_classes, num_classes]
        
        # 应用注意力权重
        attended_embeddings = torch.matmul(attention_weights, embedded)  # [1, num_classes, embedding_dim]
        
        # 移除模拟的batch_size维度，以返回原始的期望输出形状
        attended_embeddings = attended_embeddings.squeeze(0)  # [num_classes, embedding_dim]
        
        return attended_embeddings

def get_type_hidden(type_raw):
    # 参数设置
    num_classes = 5  # 嵌入的维度，这里应该是指嵌入的词汇数量
    embedding_dim = 512  # 创建嵌入层的维度

    # 创建模型实例
    embedding = AttentionEmbedding(num_classes, embedding_dim)

    type_raw = torch.tensor(type_raw)
    embedded = embedding(type_raw).to(device)
    return embedded


if __name__ == "__main__":
    # 准备数据
    train_data = 'classifier/data/train.jsonl'#'C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/getGitData/train.jsonl'# 
    raw_data = [json.loads(line) for line in open(train_data, 'r')]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 25
    #accumulated_steps = 2

    FEATURE = open('C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/getGitData/words/features.txt').read().split('\n')
    IMPORVEMENTS = open('C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/getGitData/words/improvements.txt').read().split('\n')
    BUG_FIXES = open('C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/getGitData/words/bug_fixes.txt').read().split('\n')
    DEPRECATIONS = open('C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/getGitData/words/deprecations_removals.txt').read().split('\n')
    OTHER = open('C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/getGitData/words/other.txt').read().split('\n')
    classifier = Classifier(FEATURE, IMPORVEMENTS, BUG_FIXES, DEPRECATIONS,OTHER)

    torch.cuda.empty_cache()

    # 创建自定义Bart模型
    bart_model_name = "C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/bart-base"  # facebook/bart-large "C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/bart-large-cnn" 换模型试试
    config = BartConfig.from_pretrained(bart_model_name)
    model = CustomBartForConditionalGeneration(config).from_pretrained(bart_model_name)
    best_model_path = 'C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/model/best_model.pth'
    last_model_path = 'C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/model/last_model_bart.pth'
    best_g_model_path= 'C:/Users/Administrator/Documents/Mr.b/python本地/project/BART/model/best_g_model.pth'
    bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
    bart_model = BartModel.from_pretrained(bart_model_name)
    bartembedding = BartEmbedding(bart_model, bart_tokenizer) 
    train_data = ExampleSet(raw_data, bart_tokenizer)  # 换成bart的tokenizer试试
    dataloader = DataLoader(train_data,batch_size=batch_size,shuffle=True,collate_fn=train_data.collate_fn,num_workers=0)

    num_epochs = 1
    min_loss = float('inf')
    min_batch = 0
    total_loss = 0

    ## 设置新的G
    g_model = Graph(device)
    g_model.to(device)
    g = G(device)

    # 训练
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=4e-5)  #这里改动 5->6
    ## 设置第二个graph的optimizer
    optimizer_graph = AdamW(g_model.parameters(), lr=4e-3)
    
    model.train()  #进行训练
    for epoch in range(num_epochs):  # 进行epoch
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch['attention_mask'].to(device)

            doc = batch['doc']
            # summary = batch['summary']
            output = classifier.output_type(doc)
            if len(output) == 0:
                continue
            sent_raw = [i['commit_message'] for i in output]  # [doc1（有不一定数量的sent）,doc2,doc3,doc4...batch_size]

            sent_hidden = [bartembedding.sent_hidden_state(i) for i in sent_raw]  # [(len(doc), seq_len, hidden_dim),(),()]

            type_raw = [i['type'] for i in output]

            type_hidden = [get_type_hidden(i) for i in type_raw]

            graph = g(sent_raw, sent_hidden, type_hidden)
            sent_h = g_model(graph)

            vv = sent_h.shape
            sent_h = sent_h.unsqueeze(0)
            ss = sent_h.shape
            sent_h = sent_h.repeat(batch_size, 1, 1)  # [batch_size, 58, 517, 1024]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sent_h=sent_h,
                labels=labels)
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            optimizer_graph.zero_grad()
            
            loss.backward()
           
            optimizer.step()
            optimizer_graph.step()
        
            # 输出每个批次的损失值
            print(
                f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item()}'
            )

            # 保存最低损失的模型
            if len(dataloader) - batch_idx < 100 and loss.item() < min_loss:  # and loss.item() < 1.5
                min_loss = loss.item()
                min_batch = batch_idx
                torch.save(model.state_dict(), best_model_path)
                ## 设置graph保存模型
                torch.save(g_model.state_dict(), best_g_model_path)
                print(
                    f'New minimum loss: {min_loss}, model saved to {best_model_path}'
                )

            if len(dataloader) - batch_idx < 3:
                break

    print('min_loss: ', min_loss, ' min_batch: ', min_batch)
    torch.save(model.state_dict(), last_model_path)
    print(f'Training completed. Last model saved')
