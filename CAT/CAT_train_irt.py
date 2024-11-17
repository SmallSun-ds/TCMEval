import json
from CAT.utils.logger import setuplogger
import pandas as pd
from model.IRT import IRTModel
from CAT.model.dataset.train_dataset import TrainDataset

setuplogger()

# choose dataset here
dataset = 'resident'
# choose concept_map here
concept_map = 'resident_concept_map'
# modify config here
config = {
    'learning_rate': 0.002,
    'batch_size': 256,
    'num_epochs': 200,
    'num_dim': 1,  # for IRT or MIRT
    'device': 'cuda',
}

# read datasets
train_triplets = pd.read_csv(f'./data/{dataset}.csv', encoding='utf-8').to_records(index=False)

concept_map = json.load(open(f'./concept_map/{concept_map}.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}

train_data = TrainDataset(train_triplets, concept_map,
                          10,
                          1836,
                          8)

# define model here
model = IRTModel(**config)
# train model
model.init_model(train_data)
model.train(train_data, log_step=10)

# save model
model.adaptest_save('./save/irt.pt')
