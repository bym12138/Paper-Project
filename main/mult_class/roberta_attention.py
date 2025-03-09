from transformers import RobertaModel, RobertaTokenizer
import torch
import torch.nn as nn

class CustomRobertaModel(nn.Module):
    def __init__(self, model):
        super(CustomRobertaModel, self).__init__()
        self.roberta = model
        self.hidden_size = self.roberta.config.hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8).to(self.device)
        self.linear_a = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)

    def forward(self, input_ids, attention_mask, vector_a):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        vector_a_transformed = self.linear_a(vector_a).unsqueeze(1)  # (batch_size, 1, hidden_size)
        combined_states = torch.cat([vector_a_transformed, hidden_states], dim=1)  # (batch_size, seq_length+1, hidden_size)
        attn_output, _ = self.attention(combined_states, combined_states, combined_states)
        cls_token_output = torch.mean(attn_output, dim=1) 
        a=cls_token_output.shape
        return cls_token_output

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('mult_class/roberta-base')
    model = CustomRobertaModel()

    # 示例输入
    inputs = tokenizer("Hello", return_tensors="pt")
    vector_a = torch.rand(1,768)

    # 前向传播
    logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], vector_a=vector_a)
