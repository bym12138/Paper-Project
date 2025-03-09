import torch
import pandas as pd
from dgl_graph import G
from GCN import GCN
from roberta_attention import CustomRobertaModel
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel,RobertaForMaskedLM
from fusionnet import FusionNet
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentenceEmb:
    def __init__(self, model, tokenizer, device=None):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def mask_type(self, text):
        start_phrase = "this is type of "
        start_index = text.find(start_phrase)
        if start_index != -1:
            start_index += len(start_phrase)
            masked_text = text[:start_index] + "<mask>"
        else:
            masked_text = text
        return masked_text

    def _tokenizer(self,masked_text,mask_phrase_length,mask_phrase_tokens):
        tokens = tokenizer.tokenize(masked_text)
        token_len = len(tokens)
        if token_len > 35:
            remaining_length = 35 - mask_phrase_length
            truncated_tokens = tokens[:remaining_length] + mask_phrase_tokens
        else:
            truncated_tokens = tokens
        final_token_ids = tokenizer.convert_tokens_to_ids(truncated_tokens)

        input_ids = torch.tensor(final_token_ids).unsqueeze(0) 
        decoded_mask_text = tokenizer.decode(input_ids.squeeze())
        attention_mask = torch.ones_like(input_ids)

        padding_length = 35 - input_ids.size(1)
        if padding_length > 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), 'constant', 0)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length), 'constant', 0)

        return input_ids, attention_mask

    def prepare_mlm_inputs(self, text, category):
        text_label = ',this is type of {}'.format(category)
        text = text + text_label
        masked_text = self.mask_type(text)

        mask_phrase = ", this is type of <mask>"
        mask_phrase_tokens = tokenizer.tokenize(mask_phrase)
        mask_phrase_length = len(mask_phrase_tokens)

        labels_phrase_tokens = tokenizer.tokenize(text_label)
        labels_phrase_length = len(labels_phrase_tokens)

        input_ids, attention_mask = self._tokenizer(masked_text,mask_phrase_length,mask_phrase_tokens)
        labels, _ = self._tokenizer(text,labels_phrase_length,labels_phrase_tokens)

        labels[labels == tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels

class CSVDataset(Dataset):
    def __init__(self, csv_file, sentence_emb, type_emb):
        self.data = pd.read_csv(csv_file)
        self.sentence_emb = sentence_emb
        self.type_emb = type_emb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data['Sentence'][idx]
        category = self.data['Category'][idx]
        input_ids, attention_mask, labels = self.sentence_emb.prepare_mlm_inputs(sentence, category)
        
        sample = {
            'input_ids':input_ids.squeeze(),
            'attention_mask':attention_mask.squeeze(),
            'labels':labels.squeeze(),
        }
        c=sample['input_ids'].shape
        d=sample['attention_mask'].shape
        e=sample['labels'].shape
        return sample

class TypeEmb:
    def __init__(self):
        self.text_types = ["New Features", "Bug Fixes", "Improvements", "Update", "Security", "Changes", "Deprecations and Removals", "Documentation and Tooling", "Miscellaneous"]
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def type_to_index(self, type_list):
        type_dict = {t: i for i, t in enumerate(type_list)}
        return type_dict

    def get_type_hidden(self, type_raw_list):
        type_dict = self.type_to_index(self.text_types)
        num_classes = len(self.text_types)
        embedding_dim = 250
  
        embedding = nn.Embedding(num_classes, embedding_dim)
        embedded_list = []
        for type_raw in type_raw_list:
            type_indices = [type_dict[t] for t in type_raw if t in type_dict]
            if type_indices: 
                type_indices_tensor = torch.tensor(type_indices, dtype=torch.long)
                embedded = embedding(type_indices_tensor)
                avg_embedding = embedded.mean(dim=0)
                embedded_list.append(avg_embedding)
            else:
                avg_embedding = torch.zeros(embedding_dim, device=self.device)
                embedded_list.append(avg_embedding)
        return torch.stack(embedded_list)


class AttentionAligner(nn.Module):
    def __init__(self, mask_token_dim, categorys_dim):
        super(AttentionAligner, self).__init__()
        self.query = nn.Linear(mask_token_dim, categorys_dim)
        self.key = nn.Linear(categorys_dim, categorys_dim)
        self.value = nn.Linear(mask_token_dim, categorys_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, mask_token_logits, categorys):
        Q = self.query(mask_token_logits)  # [batch_size, categorys_dim]
        K = self.key(categorys)  # [batch_size, categorys_dim]
        V = self.value(mask_token_logits)  # [batch_size, categorys_dim]
        Q = Q.unsqueeze(1)  # [batch_size, 1, categorys_dim]
        K = K.unsqueeze(1)  # [batch_size, 1, categorys_dim]
        V = V.unsqueeze(1)  # [batch_size, 1, categorys_dim]
        
        attention_scores = torch.bmm(Q, K.transpose(-2, -1))  # [batch_size, 1, 1]
        attention_weights = self.softmax(attention_scores)  # [batch_size, 1, 1]
        
        aligned_logits = torch.bmm(attention_weights, V)  # [batch_size, 1, categorys_dim]
        aligned_logits = aligned_logits.squeeze(1)  # [batch_size, categorys_dim]
        
        return aligned_logits

if __name__ == "__main__":
    csv_file = 'mult_class/data/train_set.csv'
    model_name = 'mult_class/roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    roberta_model = RobertaForMaskedLM.from_pretrained(model_name)
    roberta_model.to(device)
    sentence_emb = SentenceEmb(roberta_model, tokenizer) 
    type_emb = TypeEmb() 
    dataset = CSVDataset(csv_file, sentence_emb, type_emb)
    dataloader = DataLoader(dataset, batch_size=150, shuffle=True)
    optimizer = AdamW(roberta_model.parameters(), lr=2e-5, correct_bias=False)

    hidden_size = 768
    output_size = 35
    num_heads = 4
    attention_layer = AttentionAligner(hidden_size, output_size).to(device)

    num_epochs = 3
    roberta_model.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) # label

            outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



    # 保存模型
    torch.save(roberta_model.state_dict(), 'mult_class/save_model/roberta_model_mlm.pth')
    print('模型已保存到 roberta_model_mlm.pth')
