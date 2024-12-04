from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
import json

references = []
candidates = []

# Tokenize references
with open('BART/data_new/win/less_data/bart_baseline/pred.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        reference_tokens = word_tokenize(data['true_summary'])
        references.append(reference_tokens)
        candidates.append(data['generated_summary'])

meteor = meteor_score.meteor_score(references, candidates)

print(f'METEOR score: {meteor * 100:.2f}')