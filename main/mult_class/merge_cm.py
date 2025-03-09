import json
import csv
from tqdm import tqdm
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def _format(commit_message):
    pattern00=re.compile(r"\n")
    tmp = re.sub(pattern00,'',commit_message)
    pattern0=re.compile(r'\r')
    tmp = re.sub(pattern0,'',tmp)
    pattern7 = re.compile(r'\#\d+')
    tmp = re.sub(pattern7,'',tmp)
    pattern01=re.compile(r'["\"]')
    tmp = re.sub(pattern01,'',tmp)
    pattern1 = re.compile(r'^\*\*.*?\*\*')
    tmp = re.sub(pattern1,'',tmp)
    pattern2=re.compile(r'^\(.*?\)')
    tmp = re.sub(pattern2,'',tmp)
    pattern3=re.compile(r'^\[.*?\]')
    tmp = re.sub(pattern3,'',tmp)
    # pattern4=re.compile(r"^.+:\s*")  https://链接先不删
    # tmp = re.sub(pattern4,'',tmp)
    # pattern6=re.compile(r'^\*') 
    # tmp = re.sub(pattern6,'',tmp)
    pattern8=re.compile(r'\(') 
    tmp = re.sub(pattern8,'',tmp)
    pattern9=re.compile(r'\)') 
    tmp = re.sub(pattern9,'',tmp)
    pattern10=re.compile(r'\]') 
    tmp = re.sub(pattern10,',',tmp)
    pattern11=re.compile(r'\[') 
    tmp = re.sub(pattern11,',',tmp)
    format_text = tmp.strip()
    if len(format_text) > 6 and len(format_text.split()) > 2:
        return format_text
    return ''

def __clean_data(text): 
    doc = nlp(text)
    new_text = " ".join([token.text for token in doc if token.pos_ not in ["PUNCT","SYM","NUM","SPCAE","X"]])
    text = _format(new_text)
    doc = nlp(text)
    new_text = " ".join([token.text for token in doc if token.pos_ not in ["PUNCT","SYM","NUM","SPCAE","X"]])
    return new_text

'''
'Move CONTRIBUTING and add PULL_REQUEST_TEMPLATE (#1017)\n\n* Move CONTRIBUTING to top level for increased visibility\r\n\r\n* Add PR template to guide contributors\r\n\r\n* Oops, fix name'
'Move CONTRIBUTING and add \n\n Move CONTRIBUTING to top level for increased visibility \r\n\r\n Add PR template to guide contributors \r\n\r\n Oops fix name'
'Move CONTRIBUTING and add'
'''


input_file = 'mult_class/data/rnsum.jsonl'
output_file = 'mult_class/data/rn.txt'
all_sentences = []
format_all_sentences = []


from tqdm import tqdm
import json
import csv


with open(input_file, 'r', encoding='utf-8') as infile:
    total_lines = sum(1 for line in infile)

all_sentences = []
with open(input_file, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, total=total_lines, desc="Reading lines from input file"):
        data = json.loads(line)
        commit_messages = data.get('commit_messages', [])
        all_sentences.extend(commit_messages)

format_all_sentences = list(tqdm(map(__clean_data, all_sentences), total=len(all_sentences), desc="Formatting sentences"))
with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    for sentence in tqdm(format_all_sentences, desc="Writing sentences to output file"):
        if len(sentence) > 2:
            writer.writerow([sentence])


print(f"成功提取 {len(format_all_sentences)} 个句子到 {output_file} 中。")