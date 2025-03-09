import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch.nn as nn
import json
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'noisyCleaner/roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name, local_files_only=True)
model = RobertaForMaskedLM.from_pretrained(model_name)
model.load_state_dict(torch.load('noisyCleaner/model/roberta_model_mlm.pth'))
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
        masked_text = self.mask_type(text)
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

def predict_masked_word(text):
    text += ", this is type of <mask>"
    sentence_emb = SentenceEmb(model, tokenizer, device)
    input_ids, attention_mask = sentence_emb.prepare_mlm_inputs(text)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    x = tokenizer.decode(input_ids[0])
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    mask_token_logits = logits[0, mask_token_index, :]
    predicted_token_id = torch.argmax(mask_token_logits, dim=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    return predicted_token

input_file_path = 'noisyCleaner/data/test_ori.jsonl'
output_file_path = 'noisyCleaner/data/test_type.jsonl'

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    total_lines = sum(1 for _ in open(input_file_path, 'r', encoding='utf-8'))
    for line in tqdm(infile, total=total_lines, desc="Processing lines"):
        data = json.loads(line)
        if 'commit_messages' in data:
            messages = data['commit_messages']
            data['commit_messages'] = [{"message": msg, "type": predict_masked_word(msg)} for msg in messages]
        data = {
            "commit_messages": data.get('commit_messages', []),
            "release_note": data.get('release_note', {})
        }
        json.dump(data, outfile)
        outfile.write('\n')
