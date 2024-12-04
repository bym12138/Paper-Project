import jsonlines

# 输入的JSONL文件路径
jsonl_file = "/data/home2/kjb/bym/project/BART/data/pred_图和bart_rouge:29.63.jsonl"#"/data/home2/kjb/bym/project/BART/data/pred_纯bart_rouge:29.40.jsonl"

# 输出的候选摘要文件路径
candidate_file = "/data/home2/kjb/bym/project/BART/data/test_rouge/candidate/001_candidate.txt"

# 输出的参考摘要文件路径
reference_file = "/data/home2/kjb/bym/project/BART/data/test_rouge/reference/001_reference.txt"

# 打开JSONL文件并逐行读取
with jsonlines.open(jsonl_file) as reader:
    for obj in reader:
        # 提取generated_summary和true_summary的值
        generated_summary = obj["generated_summary"]
        true_summary = obj["true_summary"]
        
        # 将generated_summary保存到候选摘要文件
        with open(candidate_file, "a") as candidate:
            candidate.write(generated_summary + "\n")
        
        # 将true_summary保存到参考摘要文件
        with open(reference_file, "a") as reference:
            reference.write(true_summary + "\n")
