#源gatlayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

class GATLayer(nn.Module):
    def __init__(self, out_dim):
        super(GATLayer, self).__init__()
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.combination = nn.Linear(2 * out_dim, out_dim)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'sent': edges.src['h'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['sent'], dim=1)
        return {'h': h}

    def forward(self, g, h_sent, h_type):
        a=h_sent.shape
        b=h_type.shape
        g.nodes['sent'].data['h'] = h_sent
        g.nodes['type'].data['h'] = h_type
    
        g.apply_edges(self.edge_attention, etype='has')
        g.update_all(self.message_func, self.reduce_func, etype='has')
        
        return g.ndata['h']    

#多层神经网络
class MultiLayerGAT(nn.Module):
    def __init__(self, hidden_dim=768, out_dim=768, num_layers=3):
        super(MultiLayerGAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden_dim))
        self.layers.append(GATLayer(out_dim))

    def forward(self, g, h_sent, h_type):
        for i, layer in enumerate(self.layers):
            if i == 0:
                h = layer(g, h_sent, h_type)
            else:
                h = layer(g, h['sent'], h['type'])
        return h


class MultiHeadGATLayer(nn.Module):
    def __init__(self, out_dim, num_heads):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        self.attn_fc = nn.ModuleList([nn.Linear(2 * self.head_dim, 1, bias=False) for _ in range(num_heads)])
        self.combination = nn.Linear(out_dim, out_dim)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)  # Concatenate source and destination node features
        z2 = z2.view(-1, self.num_heads, 2 * self.head_dim)  # Reshape for multi-head attention
        a = [F.leaky_relu(self.attn_fc[i](z2[:, i, :])) for i in range(self.num_heads)]
        return {'e': torch.stack(a, dim=1)}

    def message_func(self, edges):
        return {'sent': edges.src['h'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['sent'].view(nodes.mailbox['sent'].shape[0], nodes.mailbox['sent'].shape[1], self.num_heads, self.head_dim), dim=1)
        h = h.view(h.shape[0], -1)
        return {'h': self.combination(h)}

    def forward(self, g, h_sent, h_type):
        h_sent = h_sent.view(-1, self.num_heads, self.head_dim)
        h_type = h_type.view(-1, self.num_heads, self.head_dim)
        g.nodes['sent'].data['h'] = h_sent
        g.nodes['type'].data['h'] = h_type
        g.apply_edges(self.edge_attention, etype='has')
        g.update_all(self.message_func, self.reduce_func, etype='has')
        return {'sent': g.nodes['sent'].data['h'].view(-1, self.out_dim), 'type': g.nodes['type'].data['h'].view(-1, self.out_dim)}

class MultiLayerGAT2(nn.Module):
    def __init__(self, hidden_dim=768, out_dim=768, num_heads=2, num_layers=3):
        super(MultiLayerGAT2, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MultiHeadGATLayer(hidden_dim, num_heads))  # Input layer
        for _ in range(num_layers - 2):
            self.layers.append(MultiHeadGATLayer(hidden_dim, num_heads))  # Hidden layers
        self.layers.append(MultiHeadGATLayer(out_dim, num_heads))  # Output layer

    def forward(self, g, h_sent, h_type):
        h = {'sent': h_sent, 'type': h_type}
        for i, layer in enumerate(self.layers):
            h = layer(g, h['sent'], h['type'])
        return h