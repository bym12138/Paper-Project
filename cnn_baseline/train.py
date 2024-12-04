# embedding 改了
from transformers import BertTokenizer, BertModel, BartConfig, BartTokenizer, BartModel
import torch
import torch.nn as nn

import dgl
from torch.utils.data import DataLoader
from torch.optim import AdamW

import json
from classifier import Classifier
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import Adafactor, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from create_cnn import CNN_Model
from custom_modeling_bart import CustomBartForConditionalGeneration
from torch.optim.lr_scheduler import StepLR

class BertEmbedding():
    def __init__(self, model,tokenizer):
        self.tokenizer = tokenizer
        self.model = model.to(device)
    
    # 使用bart编码器
    def sent_hidden_state(self, doc):
        with torch.no_grad():  # 禁用梯度计算以减少内存使用
            input_ids_list = [self.tokenizer.encode(sentence, add_special_tokens=True, padding="max_length", truncation=True, return_tensors='pt') for sentence in doc if sentence]
            embeddings = []
            for input_ids in input_ids_list:
                input_ids = input_ids.to(device)
                outputs = self.model(input_ids)
                embedding = outputs.last_hidden_state[:, 0, :]
                # shape_embedding = embedding.shape
                
                #增加一个1024->512维度
                linnet_sent = nn.Linear(768, 512, bias=False).to(device)
                embedding = linnet_sent(embedding)  
                
                
                embeddings.append(embedding)


        
        # 将张量列表转换为张量
        embeddings = torch.cat(embeddings, dim=0)
        shape = embeddings.shape # (4,512) 4是4个len(doc)

        return embeddings
 
        
class ExampleSet(Dataset):
    def __init__(self, jsonl ,tokenizer):
        self.data = self.load_data(jsonl)
        self.tokenizer = tokenizer
        
    
    def load_data(self, jsonl):
        Data = {}
        i = 0
        for idx, item in enumerate(jsonl):
            # 循环其中一条messages，一个条组doc
            doc = item['single_commit_messages']
            summary = item['single_releases_note']

            Data[i] = {
                'doc': doc,
                'summary': summary 
            }
            i += 1
        return Data
     
    def collate_fn(self,batch_samples):
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            batch_inputs.append(sample['doc'])
            batch_targets.append(sample['summary'])
       
        input_ids,labels = self.encode(batch_inputs, batch_targets)
        
        sent_list = [batch_input.split(',') for batch_input in batch_inputs]
       
        batch_data = dict()
        batch_data['input_ids'] = input_ids['input_ids']
        batch_data['labels'] = labels['input_ids']
        batch_data['attention_mask'] = input_ids['attention_mask']
        batch_data['doc'] = sent_list
        # batch_data['summary'] = batch_targets
        
        return batch_data
    
        
    def encode(self, batch_inputs, batch_targets):
        input_ids = self.tokenizer(
            batch_inputs, 
            padding="max_length", 
            max_length=256,
            truncation=True, 
            return_tensors="pt"
        )
      
        labels = self.tokenizer(
            batch_targets, 
            padding="max_length", 
            max_length=128,
            truncation=True, 
            return_tensors="pt"
        )     

        return (input_ids, labels)
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
     
def get_type_hidden(type_raw):
    # type_raw为类别id，在此函数修改type的向量化形式 [0,4,1,2,2] len(doc)
    # # 换另一种方法
    num_classes = 5
    # 嵌入的维度
    embedding_dim = 64
    # 创建嵌入层
    embedding = nn.Embedding(num_classes, embedding_dim)
    type_raw = torch.tensor(type_raw)
    embedded = embedding(type_raw).to(device)
    
    a = embedded.shape # (4,64) (len(doc),hidden_dim)
    return embedded


if __name__ == "__main__":
    # 准备数据
    dir = 'classifier/data/train.jsonl'
    raw_data = [json.loads(line) for line in open(dir, 'r')]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20
    accumulated_steps = 2

    FEATURE = open('BART/getGitData/words/features.txt').read().split('\n')
    IMPORVEMENTS = open('BART/getGitData/words/improvements.txt').read().split('\n')
    BUG_FIXES = open('BART/getGitData/words/bug_fixes.txt').read().split('\n')
    DEPRECATIONS = open('BART/getGitData/words/deprecations_removals.txt').read().split('\n')
    OTHER = open('BART/getGitData/words/other.txt').read().split('\n')
    classifier = Classifier(FEATURE,IMPORVEMENTS,BUG_FIXES,DEPRECATIONS,OTHER)
    
    torch.cuda.empty_cache()
    
    
    # 创建自定义Bart模型
    bart_model_name = "BART/bart-base" # "BART/bart-large-cnn" 换模型试试
    config = BartConfig.from_pretrained(bart_model_name)
    model = CustomBartForConditionalGeneration(config).from_pretrained(bart_model_name)
    best_model_path = 'BART/model/best_model.pth'
    # last_model_path = 'BART/model/last_model.pth'
    
    bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name) 
    bart_model = BartModel.from_pretrained(bart_model_name)  

    train_data = ExampleSet(raw_data,bart_tokenizer) # 换成bart的tokenizer
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn, num_workers=0)

    num_epochs = 1
    min_loss = float('inf')
    min_batch = 0
    total_loss = 0
    
    # 训练
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=4e-5) # 4e-6
    
    
    bertembedding = BertEmbedding(bart_model,bart_tokenizer) # 实际上是bart了
    
    model.train()   #进行训练
    for epoch in range(num_epochs):  # 进行epoch
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch['attention_mask'].to(device)
                        
            doc = batch['doc']
            output = classifier.output_type(doc)
            if len(output) == 0:
                continue
            sent_raw = [i['commit_message'] for i in output] # [doc1（有不一定数量的sent）,doc2,doc3,doc4...batch_size]         
            
            sent_hidden = [bertembedding.sent_hidden_state(i) for i in sent_raw] # [(len(doc), seq_len, hidden_dim),(),()]
            
            type_raw = [i['type'] for i in output]
            
            type_hidden = [ get_type_hidden(i) for i in type_raw]           
            cnn = CNN_Model(device=device) #保证输入的维度和cnn创建的维度，如何和batch_size和length和维度对上
            sent_h = cnn(sent_hidden, type_hidden) # 保证输出维度和graph的维度一致

          
            vv = sent_h.shape
            sent_h = sent_h.unsqueeze(0)
            ss = sent_h.shape
            sent_h = sent_h.repeat(batch_size, 1, 1) # [batch_size, 58, 517, 1024]
           
            
            
           
            
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            # decoder_input_ids=labels,
                            # decoder_attention_mask=decoder_attention_mask,
                            sent_h=sent_h,
                            labels=labels)
            loss = outputs.loss
            
            
            (loss/accumulated_steps).backward()
            if (batch_idx + 1 ) % accumulated_steps == 0:
                optimizer.step() 
                optimizer.zero_grad()
            total_loss += loss.item()
            
            
            # loss.requires_grad_(True)   #让 backward 可以追踪这个参数并且计算它的梯度
            # # print(loss)
            
            # accelerator.backward(loss)  #替代loss.backward()，进行反向传播，计算当前梯度
            # optim.step()    #更新模型参数
            # scheduler.step()    #更新学习率
            # optim.zero_grad()   #清空之前的梯度
           
            
            # 输出每个批次的损失值
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item()}')

            # 保存最低损失的模型
            if len(dataloader) - batch_idx < 100 and loss.item() < min_loss: # and loss.item() < 1.5 
                min_loss = loss.item()
                min_batch = batch_idx
                torch.save(model.state_dict(), best_model_path)
                print(f'New minimum loss: {min_loss}, model saved to {best_model_path}')
            
            if len(dataloader) - batch_idx < 3:
                break
                
            # delete caches
            del input_ids, labels, attention_mask, #decoder_attention_mask, doc, output, sent_raw, sent_hidden, type_raw, type_hidden, graph, sent_h, outputs, loss
            torch.cuda.empty_cache()
    print('min_loss: ',min_loss,' min_batch: ',min_batch)
    # torch.save(model.state_dict(), last_model_path)
    print(f'Training completed. Last model saved')