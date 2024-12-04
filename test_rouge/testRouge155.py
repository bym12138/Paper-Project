from pyrouge import Rouge155
import datetime
import os
import shutil
import re
from logger import *
import jsonlines
from itertools import islice


pred_jsonl = 'BART/data/pred.jsonl'

_PYROUGE_TEMP_FILE = "BART/test_rouge/"
 

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}", 
        "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'} 

def clean(x): 
    if isinstance(x, list):
        x = x[0]
    x = x.lower()
    return re.sub( 
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", 
            lambda m: REMAP.get(m.group()), x)

# REPLACEMENTS = {
#     '<BugFix>': '',
#     '</BugFix>': '',
#     '<Features>': '',
#     '</Features>': '',
#     '<Improvements>': '',
#     '</Improvements>': '',
#     '<Deprecations>': '',
#     '</Deprecations>': ''
# }

# def clean(x):
#     if isinstance(x, list):
#         x = x[0]
#     for old, new in REPLACEMENTS.items():
#         x = x.replace(old, new)
#     x = x.lower()
#     if len(x) < 3:
#         return ''
#     else:
#         return re.sub( 
#             r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", 
#             lambda m: REMAP.get(m.group()), x)
    

def pyrouge_score_all(hyps_list, refer_list, remap = True):
    PYROUGE_ROOT = 'BART/test_rouge/temp'
    SYSTEM_PATH = 'BART/test_rouge/temp/result'
    MODEL_PATH = 'BART/test_rouge/temp/gold'
    if os.path.exists(SYSTEM_PATH):
        shutil.rmtree(SYSTEM_PATH)
    os.makedirs(SYSTEM_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)
    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        model_file = os.path.join(MODEL_PATH, 'Reference.A.%d.txt' % i)

        refer = clean(refer_list[i]) if remap else refer_list[i]
        hyps = clean(hyps_list[i]) if remap else hyps_list[i]
    
        with open(system_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))
        with open(model_file, 'wb') as f:
            f.write(refer.encode('utf-8'))

    r = Rouge155()

    r.system_dir = 'BART/test_rouge/temp/result'#SYSTEM_PATH
    r.model_dir = 'BART/test_rouge/temp/gold'
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Reference.[A-Z].#ID#.txt'

    #try:
    output = r.convert_and_evaluate()
    print(output)
    output_dict = r.output_to_dict(output)
    # except Exception as e:
    #     print('出现报错',e) #出现报错，output_dict没有赋值，然后第70行local variable 'output_dict' referenced before assignment
    # finally:
    logger.error("[FINALLY], delete PYROUGE_ROOT...")
    shutil.rmtree(PYROUGE_ROOT)

    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'] = output_dict['rouge_1_precision'], output_dict['rouge_1_recall'], output_dict['rouge_1_f_score']
    scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'] = output_dict['rouge_2_precision'], output_dict['rouge_2_recall'], output_dict['rouge_2_f_score']
    scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'] = output_dict['rouge_l_precision'], output_dict['rouge_l_recall'], output_dict['rouge_l_f_score']
    print(scores) 
    return scores

def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
 
    return False

def get_sent_list():
    # 定义存储生成摘要和真实摘要的列表
    hyps_list = []
    refer_list = []

    # 打开jsonl文件
    with jsonlines.open(pred_jsonl) as reader:
        # 逐行读取文件内容
    
        # 使用islice仅迭代第1000行到第2000行
        for obj in reader:#islice(reader, 3590, 3600):
            # 提取生成的摘要和真实摘要
            generated_summary = obj['generated_summary']
            true_summary = obj['true_summary']
            if is_chinese(true_summary):
                continue
            if len(generated_summary) < 20 or len(true_summary) < 20:
                continue
            # 将摘要添加到相应的列表中
            hyps_list.append(generated_summary)
            refer_list.append(true_summary)

    # 打印生成的摘要列表和真实摘要列表
    print("Generated Summaries:")
    print(len(hyps_list))
    print("\nTrue Summaries:")
    print(len(refer_list))
    pyrouge_score_all(hyps_list, refer_list)

get_sent_list()