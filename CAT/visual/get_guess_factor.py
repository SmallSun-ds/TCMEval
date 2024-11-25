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

# question_id = 1000
# question_guess_factor = model.get_gamma(question_id)
#
# print(f'question {question_id} guess_factor: {question_guess_factor}')
# print("type of question_guess_factor: ", type(question_guess_factor))
# print("shape of question_guess_factor: ", question_guess_factor.shape)

guess_factor_list = []
for i in range(num_questions):
    question_guess_factor = model.get_gamma(i)
    guess_factor_list.append(question_guess_factor)

# 分析guess_factor的分布
guess_factor_list = np.array(guess_factor_list)
print("shape of guess_factor_list: ", guess_factor_list.shape)

plt.figure()
plt.hist(guess_factor_list, bins=100, alpha=0.75)
plt.xlabel('guess_factor')
plt.ylabel('count')
plt.title('guess_factor distribution')
plt.show()
