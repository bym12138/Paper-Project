import json
from collections import Counter

def count_scores(jsonl_file):
    score_counter = Counter()

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # 如果是空行，直接跳过
                continue
            try:
                data = json.loads(line)
            except json.decoder.JSONDecodeError:
                # 如果这一行不是合法的 JSON，直接跳过
                continue

            commit_messages = data.get("commit_messages", [])
            for commit in commit_messages:
                score = commit.get("score")
                # 如果 score 有效且只在 [1, 2, 3] 范围内
                if score in [1, 2, 3]:
                    score_counter[score] += 1

    return score_counter

if __name__ == "__main__":
    file_path = "noisyCleaner/data/train_type_score_copy.jsonl"
    result = count_scores(file_path)

    score_1_count = result[1]
    score_2_count = result[2]
    score_3_count = result[3]

    print("score=1 的数量:", score_1_count)
    print("score=2 的数量:", score_2_count)
    print("score=3 的数量:", score_3_count)

    total_count = score_1_count + score_2_count + score_3_count
    print("总的数量:", total_count)

    if total_count > 0:
        print("score=1 占比: {:.2f}%".format(score_1_count / total_count * 100))
        print("score=2 占比: {:.2f}%".format(score_2_count / total_count * 100))
        print("score=3 占比: {:.2f}%".format(score_3_count / total_count * 100))
    else:
        print("总数量为 0，无法计算占比。")
