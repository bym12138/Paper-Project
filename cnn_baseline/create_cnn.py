import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class CNN_Model(nn.Module):
    def __init__(self, device):
        super(CNN_Model, self).__init__()
        # 设定卷积层参数
        self.conv_sentence = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1).to(device)
        self.conv_type = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, padding=1).to(device)
        self.conv_final = nn.Conv1d(in_channels=512, out_channels=768, kernel_size=3, padding=1).to(device)

    def forward(self, sentence_input, type_input):
        # 适应CNN的输入维度：batch_size, channels, length
        sentence_input = pad_sequence(sentence_input, batch_first=True)
        type_input = pad_sequence(type_input, batch_first=True)
        a = sentence_input.shape #(1,2,512) batch_size,len(doc),dim
        b = type_input.shape #(1,2,64)
        sentence_input = sentence_input.permute(0, 2, 1)
        type_input = type_input.permute(0, 2, 1)
        
        # 应用卷积和激活函数
        conv_out_sentence = F.relu(self.conv_sentence(sentence_input))
        conv_out_type = F.relu(self.conv_type(type_input))

        # 使用全局最大池化来获取最重要的特征
        pooled_sentence = F.max_pool1d(conv_out_sentence, kernel_size=conv_out_sentence.size(2)).squeeze(2)
        pooled_type = F.max_pool1d(conv_out_type, kernel_size=conv_out_type.size(2)).squeeze(2)

        # 结合两种类型的特征
        concatenated = torch.cat((pooled_sentence, pooled_type), dim=1)

        # 最后的卷积层处理结合后的特征
        final_conv_input = concatenated.unsqueeze(2)  # 增加一个维度以适应卷积层
        final_output = F.relu(self.conv_final(final_conv_input))
        final_pooled = F.max_pool1d(final_output, kernel_size=final_output.size(2)).squeeze(2)

        return final_pooled

# 示例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_Model(device)