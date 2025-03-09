import torch
import torch.nn as nn

class MultiPromptModel(nn.Module):
    def __init__(self, num_templates, d_joint=512, lambda_reg=0.01):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_templates, d_joint))  # 权重矩阵
        self.lambda_reg = lambda_reg

    def forward(self, h_joint):
        # 计算模板权重
        weights = torch.softmax(torch.matmul(h_joint, self.W.T), dim=-1)  # [batch, n]
        return weights

    def loss(self, logits_list, labels, weights):
        # 加权交叉熵损失
        loss_ce = 0
        for i in range(len(logits_list)):
            ce = nn.CrossEntropyLoss(reduction='none')(logits_list[i], labels)
            loss_ce += (weights[:, i] * ce).mean()
        
        # L2正则化（权重矩阵W）
        reg = torch.norm(self.W, p=2) ** 2
        total_loss = loss_ce + self.lambda_reg * reg
        return total_loss