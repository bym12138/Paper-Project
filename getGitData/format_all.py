import json
from tqdm import tqdm
import spacy
import re

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


def _format(commit_message):
    pattern00=re.compile(r"\n.*")
    tmp = re.sub(pattern00,'',commit_message)
    pattern0=re.compile(r'\r')
    tmp = re.sub(pattern0,'',tmp)
    pattern01=re.compile(r'["\"]')
    tmp = re.sub(pattern01,'',tmp)
    pattern1 = re.compile(r'^\*\*.*?\*\*')
    tmp = re.sub(pattern1,'',tmp)
    pattern2=re.compile(r'^\(.*?\)')
    tmp = re.sub(pattern2,'',tmp)
    pattern3=re.compile(r'^\[.*?\]')
    tmp = re.sub(pattern3,'',tmp)
    pattern4=re.compile(r"^.+:\s*")  
    tmp = re.sub(pattern4,'',tmp)
    pattern5=re.compile(r'^\*')  
    tmp = re.sub(pattern5,'',tmp)
    pattern5=re.compile(r'^\*')  
    tmp = re.sub(pattern5,'',tmp)
    format_text = tmp.strip()
    return format_text

def remove_custom_tokens(text):
    unwanted_pos = ["PUNCT","PROPN","SYM","NUM","SPCAE","X"] 
    # 遍历所有的token
    for token in text:
        # 如果token的词性标记在unwanted_pos列表中，并且满足其他条件，就将该token设置为None
        if token.pos_ in unwanted_pos and (token.like_url or token.is_stop or token.like_num):
            token.is_parsed = False
            token.is_stop = True
            
    text = _format(text)
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
    
        
def clean_release_note(release_note):
    lists = []
    for key, value in release_note.items():
        format_value = list(map(__clean_data, value))
        if len(format_value):
            lists.extend(format_value)
    return ','.join(lists)


# 原commit格式化，release没，但是single都格式化了
def process(idx, item):
    obj = []
    commit_messages = item['commit_messages']
    commit_messages_f = list(map(__clean_data,commit_messages))
    single_commit_messages_f = ','.join(commit_messages_f)
    release_note = item['release_note']
    single_releases_note_f = clean_release_note(release_note)
    item['single_commit_messages'] = single_commit_messages_f
    item['single_releases_note'] = single_releases_note_f
    with open('classifier/data/rouge_b.jsonl', 'a') as f: 
        json.dump(item, f)
        f.write("\n")
    


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    raw_data = [json.loads(line) for line in open('classifier/data/all.jsonl','r')]
    for idx, item in enumerate(tqdm(raw_data[:25324])):
        process(idx, item) 