

import json
from enum import Enum, unique
import re
from tqdm import tqdm
'''不应该基于规则，应该根据bert分类器做'''
# dir = '/data/home2/kjb/bym/project/classifier/'

# FEATURE = open(dir+'words/features.txt').read().split('\n')
# IMPORVEMENTS = open(dir+'words/improvements.txt').read().split('\n')
# BUG_FIXES = open(dir+'words/bug_fixes.txt').read().split('\n')
# DEPRECATIONS = open(dir+'words/deprecations_removals.txt').read().split('\n')
# OTHER = open(dir+'words/other.txt').read().split('\n')

@unique
class Type(Enum):
    features = 0
    improvements = 1
    bug_fixes = 2
    deprecations_removals = 3
    other = 4


class Classifier:
    def __init__(self, FEATURE,IMPORVEMENTS,BUG_FIXES,DEPRECATIONS,OTHER):
        self.classify_data = {
            "commit_message":[],
            "type":[]
        }
        self.FEATURE = FEATURE
        self.IMPORVEMENTS = IMPORVEMENTS
        self.BUG_FIXES = BUG_FIXES
        self.DEPRECATIONS = DEPRECATIONS
        self.OTHER = OTHER

    def _format(self, commit_message):
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


    def classifier(self, commit_message):     
        if commit_message == '' or len(commit_message) < 10 or len(commit_message.split(' '))<=2:
            return 0
        m = commit_message[0:10].lower()
        type = None
        # print('message:',m)
        if any(x in m for x in self.FEATURE):
            type = Type.features
        elif any(x in m for x in self.IMPORVEMENTS):
            type = Type.improvements
        elif any(x in m for x in self.BUG_FIXES):
            type = Type.bug_fixes
        elif any(x in m for x in self.DEPRECATIONS):
            type = Type.deprecations_removals
        # elif any(x in m for x in self.OTHER):
        #     type = Type.other
        else:
            type = None  
            #type = Type.other # 不要other试试

        if type == None:
            return 0
        commit_message = self._format(commit_message)
        if commit_message == '' or len(commit_message) < 5 or len(commit_message.split(' '))<=2:
            return 0
        self.classify_data['type'].extend([type.value])
        self.classify_data['commit_message'].extend([commit_message])
        
        return 1 # 只有部分commit_message是有type的

    def sent_type(self, messages):    
        a = filter(self.classifier, messages)
        b = list(a) # 筛选出只有类型的句子
        return self.classify_data # 计算出每个句子对用的类型 (9,5) (type_num_nodes, type_node_feat_dim)

    def output_type(self, messages_batch):
        data = []
        for messages in messages_batch:
            classify_data = self.sent_type(messages)
            if len(classify_data['type']):
                data.append(classify_data)
                self.classify_data = {
                    "commit_message":[],
                    "type":[]
                }
        return data
