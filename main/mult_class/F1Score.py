import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


file_path = "mult_class/data/class_with_predictions.csv"# "mult_class/data/class_with_predictions_500.csv"

data = pd.read_csv(file_path)

true_labels = data['Category']
predicted_labels = data['Predict']

# 计算各项指标
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
accuracy = accuracy_score(true_labels, predicted_labels)

# 打印结果
results = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
    'Score': [precision, recall, f1, accuracy]
})

# 显示结果
print(results)
