import json
from CAT.utils.logger import setuplogger
import pandas as pd
from model.IRT import IRTModel
from CAT.model.dataset.train_dataset import TrainDataset

setuplogger()
import wandb
# 初始化 W&B
wandb.init(
    project="IRT",  # 项目名称
    name="CAT_Train",  # 实验名称
)

index = 0
setting_info = [
    {"task": "resident", "dataset": "resident_train", "num_students": 10, "num_questions": 1836},
]

config = {
    'learning_rate': 0.002,
    'batch_size': 256,
    'num_epochs': 200,
    'num_dim': 8,  # IRT if num_dim == 1 else MIRT
    'device': 'cuda',
}

task = setting_info[index]["task"]
dataset = setting_info[index]["dataset"]
num_students = setting_info[index]["num_students"]
num_questions = setting_info[index]["num_questions"]

# read datasets
train_triplets = pd.read_csv(f'./data/{dataset}.csv', encoding='utf-8').to_records(index=False)

train_data = TrainDataset(train_triplets, num_students, num_questions)

# define model here
model = IRTModel(**config)
# train model
model.init_model(train_data)
model.train(train_data, log_step=10, wandb=wandb)
wandb.finish()

# save model
model.adaptest_save(f'./save/{task}_irt.pt')
