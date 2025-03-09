import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from dgl.nn import GraphConv, GATConv
from createGraph import HybridGraph

# 混合模型：第一层使用GCN，第二层使用GAT
class GCN_GAT_Hybrid(nn.Module):
    def __init__(self, in_feats, hidden_feats_gcn, hidden_feats_gat, out_feats, num_heads):
        super(GCN_GAT_Hybrid, self).__init__()
        self.gcn_layer = GraphConv(in_feats, hidden_feats_gcn)
        self.gat_layer = GATConv(hidden_feats_gcn, hidden_feats_gat, num_heads)
        self.output_layer = nn.Linear(hidden_feats_gat * num_heads, out_feats)
    
    def forward(self, g, inputs):
        h = self.gcn_layer(g, inputs)
        h = torch.relu(h)
        h = self.gat_layer(g, h)
        h = h.flatten(1)
        h = torch.relu(h)
        h = self.output_layer(h)
        return h

# 自定义损失函数，包含类别奖励和同类惩罚机制
def custom_loss_fn(logits, labels, category_weights, same_category_penalty, train_mask):
    """
    logits: 模型预测的输出
    labels: 实际的标签
    category_weights: 不同类别的奖励权重
    same_category_penalty: 同类句子的惩罚权重
    """
    # CrossEntropy用于基本的类别分类任务
    loss_fn = nn.CrossEntropyLoss()
    
    # 计算基础损失
    base_loss = loss_fn(logits[train_mask], labels[train_mask])
    
    # 添加奖励机制：不同类别的句子获得更多的奖励（类别权重）
    # reward = sum([category_weights[label.item()] for label in labels[train_mask]]) / len(labels[train_mask])
    
    # 定义数值标签到类别名称的映射
    label_to_category = {
        0: 'Features',
        1: 'Fixes',
        2: 'Changes',
        3: 'Remove',
        4: 'Miscellaneous',
        5: 'Documentation',
        6: 'Security'
    }
    
    # 初始化奖励值为0
    reward = 0.0
    # 获取用于训练的标签，即根据 train_mask 筛选出的标签
    train_labels = labels[train_mask]
    # 获取训练集标签的数量，避免除以零的情况
    num_train_labels = len(train_labels)
    if num_train_labels == 0:
        raise ValueError("训练集中没有标签，请检查 train_mask 或标签数据是否正确。")
    # 计算奖励，确保每个类别标签都在 category_weights 中有对应的权重
    for label in train_labels:
        label_value = label.item()  # 获取标签的整数值
        # 将数值标签转换为类别名称
        if label_value not in label_to_category:
            raise KeyError(f"标签 {label_value} 未知，无法映射到类别，请检查标签定义。")
        category_name = label_to_category[label_value]  # 根据数值标签查找类别名称
        if category_name not in category_weights:
            raise KeyError(f"类别 {category_name} 不存在于 category_weights 中，请检查类别和权重定义。")  
        # 累积对应类别的权重
        reward += category_weights[category_name]
    # 计算平均奖励
    reward /= num_train_labels

    # 添加同类惩罚：如果同类的句子过多，加入惩罚
    unique_labels, label_counts = torch.unique(labels[train_mask], return_counts=True)
    penalty = sum([count.item() * same_category_penalty for count in label_counts if count.item() > 1])
    
    # 总损失 = 基础损失 - 奖励 + 同类惩罚
    total_loss = base_loss - reward + penalty
    
    return total_loss

# 训练模型
def train_hybrid_model(g, features, labels, train_mask, category_weights, num_epochs=100):
    # 定义混合模型
    in_feats = features.shape[1]
    hidden_feats_gcn = 16
    hidden_feats_gat = 8
    num_heads = 8
    out_feats = labels.max().item() + 1
    model = GCN_GAT_Hybrid(in_feats, hidden_feats_gcn, hidden_feats_gat, out_feats, num_heads)
    model = model.to(g.device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    same_category_penalty = 0.1  # 同类句子惩罚权重

    # 检查训练过程中的损失值
    for epoch in range(num_epochs):
        model.train()
        logits = model(g, features)
        
        loss = custom_loss_fn(logits, labels, category_weights, same_category_penalty, train_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model

# 生成最终发行说明，并打印每个类别的概率及其是否超过阈值
def generate_release_notes(g, model, features, commit_messages, threshold=0.3):
    """
    从模型输出中选择权重较高的句子，生成发行说明，并打印每个句子、其类别和超过阈值的概率
    """
    with torch.no_grad():
        logits = model(g, features)
        probs = torch.softmax(logits, dim=1)  # 概率分布
        
        # 假设有 9 个类别
        num_classes = probs.shape[1]
        
        # 初始化重要句子的布尔掩码
        important_sentences_info = []  # 用于存储重要句子及其相关信息
        
        # 遍历每个类别并打印其概率是否超过阈值
        for i in range(num_classes):
            class_probs = probs[:, i]  # 取出该类别的概率
            
            # 检查该类别的概率是否超过 threshold
            for idx in range(probs.shape[0]):
                #print(f'类别 {i} 概率: {class_probs[idx].item()}')

                if class_probs[idx].item() > threshold:
                    # 更新 important_sentences_info 列表，记录句子、类别和概率
                    important_sentences_info.append({
                        'sentence_idx': idx,
                        'sentence': commit_messages[idx],
                        'category': i,  # 当前类别
                        'probability': class_probs[idx].item()  # 概率
                    })

        # 打印并返回生成的发行说明
        release_notes = []
        for info in important_sentences_info:
            release_notes.append(info['sentence'])
            print(f"\n选中的句子 (索引 {info['sentence_idx']}): {info['sentence']}")
            print(f"该句子对应的类型: {info['category']}")
            print(f"该句子的概率: {info['probability']}")
            print(f"阈值为: {threshold}")

    return release_notes


# 生成最终发行说明，并打印每个类别的概率及其是否超过阈值
# def generate_release_notes(g, model, features, commit_messages, threshold=0.5):
#     """
#     从模型输出中选择权重较高的句子，生成发行说明
#     threshold: 选择句子的阈值，越大表示越严格
#     """
#     with torch.no_grad():
#         logits = model(g, features)
#         probs = torch.softmax(logits, dim=1)
        
#         # 假设有 9 个类别
#         num_classes = probs.shape[1]
        
#         # 初始化重要句子的布尔掩码
#         important_mask = torch.zeros(probs.shape[0], dtype=torch.bool, device=probs.device)
        
#         # 遍历每个类别并打印其概率是否超过阈值
#         for i in range(num_classes):
#             class_probs = probs[:, i]  # 取出该类别的概率
#             # 检查该类别的概率是否超过 threshold
#             for idx in range(probs.shape[0]):
#                 print(f'概率:{i}:{class_probs[idx].item()}')
#                 if class_probs[idx].item() > threshold:
#                     important_mask[idx] = True  # 更新掩码
#                     important_mask[idx]['prob'] = class_probs[idx].item()
#                 #     print(f"\n类别 {i} 的句子及其超过阈值的概率:")
#                 #     print(f"句子索引: {idx}, 概率: {class_probs[idx].item()} > {threshold}")

#         # 筛选出至少一个类别的概率超过阈值的句子
#         important_sentences = important_mask.nonzero(as_tuple=False).flatten()

#         # 打印生成发行说明
#         release_notes = []
#         for idx in important_sentences:
#             release_notes.append(commit_messages[idx.item()])
#             print(f"选中的句子: {commit_messages[idx.item()]}")

#     return release_notes







# 类别定义
categories = ['Features', 'Fixes', 'Features', 'Changes', 'Remove', 'Features', 'Miscellaneous', 'Documentation', 'Security']

# 定义类别到数值的映射
category_to_label = {
    'Features': 0,
    'Fixes': 1,
    'Changes': 2,
    'Remove': 3,
    'Miscellaneous': 4,
    'Documentation': 5,
    'Security': 6
}

# 类别权重，可以根据需求调整每个类别的重要性
category_weights = {
    'Features': 1.0,
    'Fixes': 1.5,
    'Update': 1.2,
    'Changes': 1.1,
    'Remove': 1.3,
    'Improvement': 1.4,
    'Miscellaneous': 0.5,
    'Documentation': 0.8,
    'Security': 2.0  # 假设Security类别的重要性较高
}

# 示例的commit messages和它们的类别
commit_messages = [
    "Added new authentication feature",
    "Fixed a critical bug in the authentication module",
    "Updated the readme file with the latest changes",
    "Changed the structure of the project files for better organization",
    "Removed deprecated API calls",
    "Improved the performance of data processing",
    "Miscellaneous clean-up in the test suite",
    "Added documentation for new API endpoints",
    "Addressed security vulnerability in login mechanism"
]

commit_categories = ['Features', 'Fixes', 'Features', 'Changes', 'Remove', 'Features', 'Miscellaneous', 'Documentation', 'Security']

# 初始化图结构类，并使用CUDA
graph_builder = HybridGraph(alpha=0.7, beta=0.3, similarity_threshold=0.7, use_cuda=True)

# 构建图
g = graph_builder.build_graph(commit_messages, commit_categories, category_weights)
# 示例数据已经在图结构中
# 提取图中的特征和标签
features = g.ndata['embedding']
labels = torch.tensor([category_to_label[category] for category in commit_categories]).to(g.device)
train_mask = torch.rand(g.num_nodes()) > 0.4  # 随机生成训练掩码

# 训练模型
hybrid_model = train_hybrid_model(g, features, labels, train_mask, category_weights)

# 生成发行说明
release_notes = generate_release_notes(g, hybrid_model, features, commit_messages)
print("生成的发行说明:", release_notes)
