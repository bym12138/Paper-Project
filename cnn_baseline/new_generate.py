from transformers import BartConfig, BartTokenizer, BartModel
import torch
from create_cnn import CNN_Model
import torch.nn as nn
import os

from custom_modeling_bart import CustomBartForConditionalGeneration
from torch.utils.data import DataLoader
from torch.optim import AdamW

import json
from classifier import Classifier
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rouge import Rouge


class BertEmbedding():
    def __init__(self, model,tokenizer):
        self.tokenizer = tokenizer
        self.model = model.to(device)
    
    def sent_hidden_state(self, doc):
        with torch.no_grad():  # 禁用梯度计算以减少内存使用
            input_ids_list = [self.tokenizer.encode(sentence, add_special_tokens=True, max_length=128, truncation=True, return_tensors='pt') for sentence in doc if sentence]
            cls_embeddings = []
            for input_ids in input_ids_list:
                input_ids = input_ids.to(device)
                outputs = self.model(input_ids)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                
                 #增加一个1024->512维度
                linnet_sent = nn.Linear(768, 512, bias=False).to(device)
                cls_embedding = linnet_sent(cls_embedding)
                
                cls_embeddings.append(cls_embedding)

        # 将张量列表转换为张量
        cls_embeddings = torch.cat(cls_embeddings, dim=0)
        shape = cls_embeddings.shape
        return cls_embeddings
       
        
class ExampleSet(Dataset):
    def __init__(self, jsonl,tokenizer):
        self.data = self.load_data(jsonl)
        self.tokenizer = tokenizer
        
    
    def load_data(self, jsonl):
        Data = {}
        i = 0
        for idx, item in enumerate(jsonl):
            # 循环其中一条messages，一个条组doc
            doc = item['single_commit_messages']
            summary = item['single_releases_note']
            # type_dim = output_type(doc)
            # data_tmp.append((doc, summary))
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
        batch_data['decoder_attention_mask'] = labels['attention_mask']
        
        batch_data['doc'] = sent_list
        # batch_data['summary'] = batch_targets
        
        return batch_data
        
    def encode(self, batch_inputs, batch_targets):
        input_ids = self.tokenizer( # input_ids,token_type_ids,attention_mask
            batch_inputs, 
            padding="max_length", 
            max_length=256,
            truncation=True, 
            return_tensors="pt"
        )#['input_ids']
        labels = self.tokenizer(
                batch_targets, 
                padding="max_length", 
                max_length=128,
                truncation=True, 
                return_tensors="pt"
        )#['input_ids']

        return (input_ids, labels)
       

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
     
def get_type_hidden(type_raw):
    num_classes = 5
    # 嵌入的维度
    embedding_dim = 64
    # 创建嵌入层
    embedding = nn.Embedding(num_classes, embedding_dim)
    type_raw = torch.tensor(type_raw)
    embedded = embedding(type_raw).to(device) # (num_type, num_classes, embedding_dim)
    return embedded

def evaluate(model, dataloader, device, bertembedding, classifier):
        model.eval()
        rouge = Rouge()
        rouge_l_scores = []
        pbar = tqdm(total=len(dataloader))
        with torch.no_grad(), open("BART/data/pred.jsonl", "w") as pred_file:
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                # decoded_string1 = tokenizer.decode(batch["input_ids"][0])
                # decoded_string2 = tokenizer.decode(batch["labels"][0])
                attention_mask = batch['attention_mask'].to(device)
                doc = batch['doc']
                output = classifier.output_type(doc)
                
                if len(output) == 0:
                    continue
                sent_raw = [i['commit_message'] for i in output]
                sent_hidden = [bertembedding.sent_hidden_state(i) for i in sent_raw]
                type_raw = [i['type'] for i in output]
                
                type_hidden = [ get_type_hidden(i) for i in type_raw]    
                
                cnn = CNN_Model(device=device)
                sent_h = cnn(sent_hidden, type_hidden)
                sent_h = sent_h.unsqueeze(0)
                sent_h = sent_h.repeat(batch_size, 1, 1)
                
	      
                # sent_h = torch.rand(batch_size,13,1024).to(device)
                
                # 使用model.generate()生成预测摘要
                generated_ids = model.generate(input_ids,
                                            attention_mask=attention_mask,
                                            max_length=128,
                                            num_beams=1,
                                            sent_h=sent_h
                                            )
                generated_summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                true_summary = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                if len(dataloader) - batch_idx < 3:
                    break
                             
                pbar.set_description('dsc')
                
                pbar.update()
           

                # 计算ROUGE-L分数
                for gen_summ, true_summ in zip(generated_summary, true_summary):
                    # if len(gen_summ) <= 5:
                    #     continue
                    if len(gen_summ) < 3 or len(true_summ) < 3:
                        continue
                    scores = rouge.get_scores(gen_summ, true_summ)
                    rouge_l_scores.append(scores[0]['rouge-l']['f'])
                    a = scores[0]['rouge-l']['f']
                    b = 1 
                    

                # 将文档、摘要和预测摘要保存到pred.jsonl文件中
                for doc_text, true_summ, gen_summ in zip(doc, true_summary, generated_summary):
                    pred_dict = {
                        "document": doc_text,
                        "true_summary": true_summ,
                        "generated_summary": gen_summ
                    }
                    pred_file.write(json.dumps(pred_dict) + "\n")
                
            pbar.close()
             
        avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
        return avg_rouge_l

if __name__ == "__main__":
    # 准备数据
    dir = 'classifier/data/test.jsonl'
    model_dir = 'BART/model/'#'BART/backup/model/'
    raw_data = [json.loads(line) for line in open(dir, 'r')]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 10

    FEATURE = open('BART/getGitData/words/features.txt').read().split('\n')
    IMPORVEMENTS = open('BART/getGitData/words/improvements.txt').read().split('\n')
    BUG_FIXES = open('BART/getGitData/words/bug_fixes.txt').read().split('\n')
    DEPRECATIONS = open('BART/getGitData/words/deprecations_removals.txt').read().split('\n')
    OTHER = open('BART/getGitData/words/other.txt').read().split('\n')
    classifier = Classifier(FEATURE,IMPORVEMENTS,BUG_FIXES,DEPRECATIONS,OTHER)
    
    # 加载自定义Bart模型
    bart_model_name = "BART/bart-base" #"facebook/bart-large" "BART/bart-large-cnn" 换模型试试
    config = BartConfig.from_pretrained(bart_model_name)
    model = CustomBartForConditionalGeneration(config).from_pretrained(bart_model_name)#('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained(bart_model_name)
    bestmodel_file = os.path.join(model_dir, 'best_model.pth')
    # bestmodel_file = os.path.join(model_dir, 'last_model.pth')
    model.load_state_dict(torch.load(bestmodel_file))#model.load_state_dict(torch.load(bestmodel_file))
    model.to(device)
    
    bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name) 
    bart_model = BartModel.from_pretrained(bart_model_name) 
    bertembedding =  BertEmbedding(bart_model,bart_tokenizer)
    
    # 加载test dataloader
    test_data = ExampleSet(raw_data, tokenizer) # 换成bart的tokenizer试试
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=test_data.collate_fn, num_workers=0)

    # 测试评估模型
    avg_rouge_l = evaluate(model, dataloader, device, bertembedding, classifier)
    print(f"Average ROUGE-L score: {avg_rouge_l}")
    

        

    
