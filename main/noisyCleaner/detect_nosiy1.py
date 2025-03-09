import json
import re
import fasttext 
from collections import Counter


def is_suspicious_word_basic(word):
    stripped = word.strip(",.!?\"'()[]{}<>#@&*:")
    if len(stripped) == 0:
        return True
    if re.search(r'([A-Za-z])\1{2,}', word):
        return True

    if re.match(r'^[^a-zA-Z]+$', word) and len(word) > 5:
        return True
    return False


# ====== 2. 基于 FastText 的辅助检测 =======
def is_suspicious_word_fasttext(word, model, sim_threshold=0.75):
    words_in_vocab = model.get_words(on_unicode_error='replace')
    if word not in words_in_vocab:
        return True
    
    neighbors = model.get_nearest_neighbors(word, k=1)
    if not neighbors: 
        return True
    
    top_sim, top_neighbor = neighbors[0]
    if top_sim < sim_threshold:
        return True
    
    return False

def detect_suspicious_words_with_fasttext(input_file, model_path, sim_threshold=0.75):
    model = fasttext.load_model(model_path)
    suspicious_words = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line_data = json.loads(line.strip())
            commit_msgs = line_data.get("commit_messages", [])
            for msg_obj in commit_msgs:
                message = msg_obj.get("message", "")
                tokens = re.findall(r"\S+", message)
                for token in tokens:
                    if is_suspicious_word_fasttext(token, model, sim_threshold=sim_threshold):
                        suspicious_words.append(token)
    
    return suspicious_words



if __name__ == "__main__":

    input_path = "test_type_score.jsonl"
    fasttext_model_path = "fasttext_model.bin" 
    similarity_threshold = 0.75
    all_suspicious = detect_suspicious_words_with_fasttext(
        input_file=input_path,
        model_path=fasttext_model_path,
        sim_threshold=similarity_threshold
    )

    counter = Counter(all_suspicious)

    print("可疑单词统计结果（出现次数由高到低）：")
    for word, freq in counter.most_common():
        print(f"{word}\t{freq}")
