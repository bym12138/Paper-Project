import json
from torchtext.data.metrics import bleu_score

references = []
candidates = []

with open('BART/data/pred.jsonl', 'r') as f: # classifier/CAS_data/CAS_single_new3.jsonl
    for line in f:
        data = json.loads(line)
        references.append([data['true_summary'].split()])
        candidates.append(data['generated_summary'].split())

bleu3 = bleu_score(candidates, references, max_n=3, weights=[1/3, 1/3, 1/3])
bleu4 = bleu_score(candidates, references, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])

print(f'BLEU-3 score: {bleu3 * 100:.2f}')
print(f'BLEU-4 score: {bleu4 * 100:.2f}')




# from nltk.translate.bleu_score import corpus_bleu
# import jsonlines



# pred_jsonl = 'BART/data_new/win/less_data/best_data/pred.jsonl'

# def get_sent_list():
#     # 定义存储生成摘要和真实摘要的列表
#     hyps_list = []
#     refer_list = []

#     # 打开jsonl文件
#     with jsonlines.open(pred_jsonl) as reader:
#         # 逐行读取文件内容
#         for obj in reader:
#             tmp = []
#             # 提取生成的摘要和真实摘要
#             generated_summary = obj['generated_summary'].split()
#             true_summary = obj['true_summary'].split()
#             tmp.append(true_summary)
#             # 将摘要添加到相应的列表中
#             hyps_list.append(generated_summary)
#             refer_list.append(tmp)



#     # 打印生成的摘要列表和真实摘要列表
#     print("Generated Summaries:")
#     print(len(hyps_list))
#     print("\nTrue Summaries:")
#     print(len(refer_list))
#     bleu_score_all(hyps_list, refer_list)
    
# def bleu_score_all(hyps_list, refer_list):
#     bleu_3_score = corpus_bleu(refer_list, hyps_list, weights=(0.33, 0.33, 0.33))
#     bleu_4_score = corpus_bleu(refer_list, hyps_list, weights=(0.25, 0.25, 0.25, 0.25))
#     print(f"BLEU-3 Score: {bleu_3_score:.4f}")
#     print(f"BLEU-4 Score: {bleu_4_score:.4f}")

# get_sent_list()








# # 计算BLEU-3评分
# bleu_3_score = sentence_bleu(reference_ngrams, candidate_ngrams, weights=(0.33, 0.33, 0.33))
# print(f"BLEU-3 Score: {bleu_3_score:.4f}")

# # 计算BLEU-4评分
# bleu_4_score = sentence_bleu(reference_ngrams, candidate_ngrams, weights=(0.25, 0.25, 0.25, 0.25))
# print(f"BLEU-4 Score: {bleu_4_score:.4f}")


# from nltk.translate.bleu_score import sentence_bleu

# # 多个参考句子
# references = [['The cat is on the mat'], ['The dog is playing in the garden']]

# # 多个候选句子
# candidates = ['The cat is on the mat', 'The dog is playing on the mat']

# # 循环计算每对参考句子和候选句子的BLEU分数
# for i, (reference, candidate) in enumerate(zip(references, candidates)):
#     # 将参考句子和候选句子转换为n-gram列表
#     reference_ngrams = [ref[0].split() for ref in reference]
#     candidate_ngrams = candidate.split()

#     # 计算BLEU-3评分
#     bleu_3_score = sentence_bleu(reference_ngrams, candidate_ngrams, weights=(0.33, 0.33, 0.33))
#     print(f"BLEU-3 Score for Sentence {i+1}: {bleu_3_score:.4f}")

#     # 计算BLEU-4评分
#     bleu_4_score = sentence_bleu(reference_ngrams, candidate_ngrams, weights=(0.25, 0.25, 0.25, 0.25))
#     print(f"BLEU-4 Score for Sentence {i+1}: {bleu_4_score:.4f}")


# bleu3 = sentence_bleu(
#     references=string_candidates,
#     hypothesis=string_target,
#     weights=(1.0 / 3, 1.0 / 3, 1.0 / 3),
# )
# return bleu3
