import torch
from config import parsers
import json
from tqdm import tqdm
import spacy


def process_classify_data():
    for idx, item in enumerate(tqdm(classify_data)):
        write_row(idx, item)

def generate_all_release_note(raw):
    lists = []
    for key, value in raw['release_note'].items():
        format_value = list(map(__clean_data, value))
        if len(format_value):
            lists.extend(format_value)
    return ','.join(lists)

def generate_all_commit_messages(raw):
    list = []
    for value in raw['commit_messages']:
        format_value = __clean_data(value)
        if len(format_value) > 5:
            list.append(format_value)
    return ','.join(list)

def clean_classify_commit_messages(raw):
    for key, value in raw['classify_commit_messages'].items():
        format_value = list(map(__clean_data, value))
        raw['classify_commit_messages'][key] = format_value
    return raw['classify_commit_messages']
 
def clean_release_note(raw):
    for key, value in raw['release_note'].items():
        format_value = list(map(__clean_data, value))
        raw['classify_commit_messages'][key] = format_value
    return raw['classify_commit_messages']


def remove_custom_tokens(text):
    unwanted_pos = ["PUNCT","PROPN","SYM","NUM","SPCAE","X"]
    # 遍历所有的token
    for token in text:
        # 如果token的词性标记在unwanted_pos列表中，并且满足其他条件，就将该token设置为None
        if token.pos_ in unwanted_pos and (token.like_url or token.is_stop or token.like_num):
            token.is_parsed = False
            token.is_stop = True
    return text

def __clean_data(text): 
    doc = nlp(text)
    new_text = " ".join([token.text for token in doc if token.pos_ not in ["PUNCT","PROPN","SYM","NUM","SPCAE","X"]])
    return new_text

def write_row(idx, classify_commit_messages):
    raw = raw_data[idx]
    raw['classify_commit_messages'] = classify_commit_messages['commit_message']
    raw['classify_commit_messages'] = clean_classify_commit_messages(raw)
    raw['single_releases_note'] = generate_all_release_note(raw)
    raw['single_commit_messages'] = generate_all_commit_messages(raw)
    raw['release_note'] = clean_release_note(raw)
    # clean [single_releases_note,single_commit_messages,classify_commit_messages,releases_note]
    with open(args.all_jsonl, 'a') as f: 
        json.dump(raw, f)
        f.write("\n")


if __name__ == '__main__':
    args = parsers()
    nlp = spacy.load("en_core_web_sm")
    classify_data = [json.loads(line) for line in open(args.classify_jsonl,'r')]
    raw_data = [json.loads(line) for line in open(args.jsonl,'r')] 
    process_classify_data()