# format all_rouge.jsonl
from rouge import Rouge
import numpy as np
import json
from tqdm import tqdm

rouge = Rouge()

data = [json.loads(line) for line in open(f"train.jsonl",'r')]
preds = []
labels = []

def wirte_row(item):
    with open('train_rouge.jsonl', 'a') as f: 
            json.dump(item, f)
            f.write("\n")

for idx,item in enumerate(tqdm(data)):
    pred = item['single_releases_note'][:1000]
    label = item['single_commit_messages'][:1000]
    
    #加入条件：single_commit_messages的条数大于2
    commit_messages_len = len(pred.split(','))
    
    if len(pred) > 5 and len(label) > 5 and commit_messages_len > 2:
        scores = rouge.get_scores(
        hyps=pred, 
        refs=label, 
        )

        rouge_l = scores[0]['rouge-l']['f']
        if rouge_l > 0.10 and len(pred) > 5:
            wirte_row(item)
print('count:',idx)