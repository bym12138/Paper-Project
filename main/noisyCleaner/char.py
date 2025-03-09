import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体为 SimHei
matplotlib.rcParams['font.family'] = 'SimHei'
# 正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False

# 模型名称
models = ['TF-IDF + SVM', '朴素贝叶斯', 'LSTM', 'GRU', 'GCN', 'GAT']
# 准确率
accuracy = [75.0, 73.5, 78.0, 77.5, 80.5, 82.5]
# F1分数
f1_score = [71.0, 69.1, 74.0, 73.5, 77.0, 79.5]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracy, width, label='准确率')
rects2 = ax.bar(x + width/2, f1_score, width, label='F1分数')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('分数 (%)')
ax.set_title('不同模型在高噪音环境下的准确率与F1分数对比')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

fig.tight_layout()
plt.show()
