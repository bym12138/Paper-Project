import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'mult_class/roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)

class TypeEmb:
    def __init__(self):
        self.text_types = ["New Features", "Bug Fixes", "Improvements", "Update", "Security", "Changes", "Deprecations and Removals", "Documentation and Tooling", "Miscellaneous"]
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def type_to_index(self, type_list):
        type_dict = {t: i for i, t in enumerate(type_list)}
        return type_dict

    def get_type_hidden(self, type_raw_list):
        type_dict = self.type_to_index(self.text_types)
        num_classes = len(self.text_types)
        embedding_dim = 250

        embedding = nn.Embedding(num_classes, embedding_dim)
        embedded_list = []
        for type_raw in type_raw_list:
            type_indices = [type_dict[t] for t in type_raw if t in type_dict]
            if type_indices:
                type_indices_tensor = torch.tensor(type_indices, dtype=torch.long)
                embedded = embedding(type_indices_tensor)
                avg_embedding = embedded.mean(dim=0)
                embedded_list.append(avg_embedding)
            else:
                avg_embedding = torch.zeros(embedding_dim, device=self.device)
                embedded_list.append(avg_embedding)
        
        return torch.stack(embedded_list)

roberta_model = RobertaModel.from_pretrained(model_name)
roberta_model.load_state_dict(torch.load('mult_class/save_model/roberta_model_mlm.pth'))
roberta_model.to(device)
roberta_model.eval()

def generate_labels(input_text, roberta_model, tokenizer, type_emb):
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=35)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
    logits = nn.Linear(sentence_embeddings.size(-1), 250).to(device)(sentence_embeddings)
    predicted_embedding = logits

    type_embeddings = type_emb.get_type_hidden([type_emb.text_types])
    similarities = torch.matmul(predicted_embedding, type_embeddings.T)
    predicted_type_idx = similarities.argmax(dim=1).item()

    predicted_label = type_emb.text_types[predicted_type_idx]

    return predicted_label

if __name__ == "__main__":
    type_emb = TypeEmb()
    input_text = "Your text here"
    generated_label = generate_labels(input_text, roberta_model, tokenizer, type_emb)
    print(f'生成的标签: {generated_label}')
