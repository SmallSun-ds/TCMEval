import json
from CAT.utils.logger import setuplogger
import pandas as pd
from model.IRT import IRTModel
from CAT.model.dataset.train_dataset import TrainDataset

setuplogger()

index = 0
setting_info = [
    {"task": "resident", "dataset": "resident_train", "concept_map": "resident_concept_map",
     "num_students": 10, "num_questions": 1836, "num_concepts": 8},
]

config = {
    'learning_rate': 0.002,
    'batch_size': 256,
    'num_epochs': 200,
    'num_dim': 1,  # for IRT or MIRT
    'device': 'cuda',
}

task = setting_info[index]["task"]
dataset = setting_info[index]["dataset"]
concept_map = setting_info[index]["concept_map"]
num_students = setting_info[index]["num_students"]
num_questions = setting_info[index]["num_questions"]
num_concepts = setting_info[index]["num_concepts"]

# read datasets
train_triplets = pd.read_csv(f'./data/{dataset}.csv', encoding='utf-8').to_records(index=False)

concept_map = json.load(open(f'./concept_map/{concept_map}.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}

train_data = TrainDataset(train_triplets, concept_map, num_students, num_questions, num_concepts)

# define model here
model = IRTModel(**config)
# train model
model.init_model(train_data)
model.train(train_data, log_step=10)

# save model
model.adaptest_save(f'./save/{task}_irt.pt')
