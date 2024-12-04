# https://huggingface.co/spaces/evaluate-metric/rouge

# blue https://pytorch.org/text/stable/index.html

import evaluate
import jsonlines


rouge = evaluate.load('rouge')
pred_jsonl = 'classifier/data/t5_new_baseline.jsonl'#'BART/data/pred.jsonl'

def get_sent_list():
    # 定义存储生成摘要和真实摘要的列表
    hyps_list = []
    refer_list = []

    # 打开jsonl文件
    with jsonlines.open(pred_jsonl) as reader:
        # 逐行读取文件内容
        for obj in reader:
            tmp = []
            # 提取生成的摘要和真实摘要
            generated_summary = obj['generated_summary']
            true_summary = obj['true_summary']
            tmp.append(true_summary)
            # 将摘要添加到相应的列表中
            hyps_list.append(generated_summary)
            refer_list.append(tmp)



    # 打印生成的摘要列表和真实摘要列表
    print("Generated Summaries:")
    print(len(hyps_list))
    print("\nTrue Summaries:")
    print(len(refer_list))
    rouge_score(hyps_list, refer_list)

def rouge_score(hyps_list, refer_list):
     results = rouge.compute(predictions=hyps_list,  references=refer_list)
     print(results)


get_sent_list()
