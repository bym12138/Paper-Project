import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'mult_class/roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name)
model.load_state_dict(torch.load('mult_class/save_model/roberta_model_mlm.pth'))
model.to(device)
model.eval() 

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

    def prepare_mlm_inputs(self, text):
        masked_text= self.mask_type(text)
        mask_phrase = ", this is type of <mask>"
        mask_phrase_tokens = tokenizer.tokenize(mask_phrase)
        mask_phrase_length = len(mask_phrase_tokens)
        tokens = tokenizer.tokenize(masked_text)
        if len(tokens) > 35:
            remaining_length = 35 - mask_phrase_length
            truncated_tokens = tokens[:remaining_length] + mask_phrase_tokens
        else:
            truncated_tokens = tokens

        final_token_ids = tokenizer.convert_tokens_to_ids(truncated_tokens)
        input_ids = torch.tensor(final_token_ids).unsqueeze(0) 
        attention_mask = torch.ones_like(input_ids)
        padding_length = 35 - input_ids.size(1)
        if padding_length > 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), 'constant', 0)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length), 'constant', 0)

        decoded_text = tokenizer.decode(input_ids.squeeze())
       
        return input_ids, attention_mask

# 定义预测函数
def predict_masked_word(text, tokenizer, model, device):
    text += ", this is type of <mask>"
    sentence_emb = SentenceEmb(model, tokenizer, device)
    input_ids, attention_mask = sentence_emb.prepare_mlm_inputs(text)
    
    # 将数据移动到设备
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    x=tokenizer.decode(input_ids[0])
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    mask_token_logits = logits[0, mask_token_index, :]
    
    # 选取概率最大的词
    predicted_token_id = torch.argmax(mask_token_logits, dim=-1)
    predicted_token = tokenizer.decode(predicted_token_id)

    return predicted_token
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('mult_class/data/test_set.csv')

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing", ncols=100):
    predict = predict_masked_word(row['Sentence'], tokenizer, model, device)
    predict = predict.strip()
    df.at[index, 'Predict'] = predict

df.to_csv('mult_class/data/class_with_predictions.csv', index=False)