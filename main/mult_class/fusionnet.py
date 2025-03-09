import torch
import torch.nn as nn

dim=768 * 2
class FusionNet(nn.Module):
    def __init__(self, input_dim=dim, output_dim=768):
        super(FusionNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(input_dim, output_dim).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.fc2 = nn.Linear(output_dim, output_dim).to(self.device)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


