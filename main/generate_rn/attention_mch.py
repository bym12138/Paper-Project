import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size).to('cuda')
        self.W2 = nn.Linear(hidden_size, hidden_size).to('cuda')
        self.V = nn.Linear(hidden_size, 1).to('cuda')

    def forward(self, query, key, value):
        key_with_weights = self.W2(key)
        query_with_weights = self.W1(query)
        query_with_weights = query_with_weights.unsqueeze(2)
        key_with_weights = key_with_weights.unsqueeze(2)
        scores = self.V(torch.tanh(query_with_weights + key_with_weights))
        attention_weights = F.softmax(scores, dim=1)
        context_vector = attention_weights * value.unsqueeze(1)
        context_vector = torch.sum(context_vector, dim=2)
        
        return context_vector, attention_weights
