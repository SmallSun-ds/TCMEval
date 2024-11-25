# num_dim = 16, pl = 3
from CAT.model.IRT import IRTModel
import pandas as pd
from CAT.model.dataset.adaptest_dataset import AdapTestDataset
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


setting_info = [
    {"task": "resident", "dataset": "resident_eval", "num_students": 9, "num_questions": 1836, 'pl': 3, 'num_dim': 16},
]

config = {
    'learning_rate': 0.002,
    'batch_size': 64,
    'num_epochs': 200,
    'device': 'cuda',
}
index = 0
task = setting_info[index]["task"]
num_dim = setting_info[index]['num_dim']
pl = setting_info[index]['pl']
dataset = setting_info[index]['dataset']
num_students = setting_info[index]['num_students']
num_questions = setting_info[index]['num_questions']

# read datasets
test_triplets = pd.read_csv(f'./data/{dataset}.csv', encoding='utf-8').to_records(index=False)

test_data = AdapTestDataset(test_triplets,
                            num_students,  # 同时为num_students个学生进行CAT测试
                            num_questions)

# checkpoint path
ckpt_path_question = f'./save/{task}_irt_dim{num_dim}_{pl}pl.pt'

model = IRTModel(**config)
model.init_model(test_data, pl=pl, num_dim=num_dim)

# 作出student_ability的数据sigmoid后的雷达图

students_ability = np.zeros((num_students, num_dim))

# students = [2, 6]  # 局部测试
# for i in range(len(students)):
#     students_id = students[i]
#     student_ability = model.get_theta(students_id)
#     students_ability[students_id] = student_ability

for i in range(num_students):  # 全量测试
    student_ability = model.get_theta(i)
    students_ability[i] = student_ability

# some trick
students_ability = (students_ability + 1) ** 2

# Radar chart setup
labels = [
    "Dimension 1", "Dimension 2", "Dimension 3", "Dimension 4",
    "Dimension 5", "Dimension 6", "Dimension 7", "Dimension 8",
    "Dimension 9", "Dimension 10", "Dimension 11", "Dimension 12",
    "Dimension 13", "Dimension 14", "Dimension 15", "Dimension 16"
]

# Apply sigmoid transformation to all student abilities
sigmoid_students_ability = sigmoid(students_ability)
# # 平方
# sigmoid_students_ability = sigmoid_students_ability ** 2

# Radar chart setup
num_students = students_ability.shape[0]
angles = np.linspace(0, 2 * np.pi, students_ability.shape[1], endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Create the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Plot each student's data
for i in range(num_students):
    student_data = np.concatenate((sigmoid_students_ability[i], [sigmoid_students_ability[i][0]]))  # Close the chart
    ax.fill(angles, student_data, alpha=0.25, label=f"Student {i + 1}")
    ax.plot(angles, student_data, linewidth=2)

# Add labels
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=8)
ax.set_title("Students Ability (Sigmoid Transformed)", va='bottom')

# Add legend (multi-column)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)

plt.tight_layout()
plt.show()
