import torch

class MultiPromptFusion(torch.nn.Module):
    def __init__(self, roberta_model, graph_model, num_templates=3, d_joint=512):
        super().__init__()
        self.roberta = roberta_model
        self.graph_model = graph_model
        self.text_proj = torch.nn.Linear(768, d_joint//2)
        self.graph_proj = torch.nn.Linear(256, d_joint//2)
        self.weight_vectors = torch.nn.Parameter(torch.randn(num_templates, d_joint))
        self.classifier = torch.nn.Linear(768, 9)  # 9个类别

    def forward(self, text_input, graph_data):
        # 文本嵌入
        text_emb = self.roberta(**text_input).last_hidden_state[:, 0, :]
        # 图嵌入
        graph_emb = self.graph_model(graph_data.x_dict, graph_data.edge_index_dict)['text']
        # 联合嵌入
        h_text = self.text_proj(text_emb)
        h_graph = self.graph_proj(graph_emb)
        h_joint = torch.cat([h_text, h_graph], dim=-1)
        # 模板权重
        weights = torch.softmax(torch.matmul(h_joint, self.weight_vectors.T), dim=-1)
        # 各模板预测
        logits_list = [self._get_template_logits(text_input, t_id) for t_id in range(num_templates)]
        # 加权融合
        final_logits = sum(w * logits for w, logits in zip(weights.T, logits_list))
        return final_logits


    