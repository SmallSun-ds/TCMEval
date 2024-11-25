import json
from CAT.utils.logger import setuplogger
import pandas as pd
from model.IRT import IRTModel
from CAT.model.dataset.train_dataset import TrainDataset
from multiprocessing import Pool
import wandb
from CAT.utils.settings import train_setting_info as setting_info

setuplogger()


def run_train(index):
    task = setting_info[index]["task"]
    dataset = setting_info[index]["dataset"]
    num_students = setting_info[index]["num_students"]
    num_questions = setting_info[index]["num_questions"]
    num_dim = setting_info[index]['num_dim']
    pl = setting_info[index]['pl']

    # 初始化 W&B
    wandb_flag = False
    if 1:
        wandb.init(
            project="IRT",  # 项目名称
            name=f"CAT_Train_{task}_dim{num_dim}_{pl}pl",  # 实验名称
        )
        wandb_flag = True

    # read datasets
    train_triplets = pd.read_csv(f'./data/{dataset}.csv', encoding='utf-8').to_records(index=False)

    train_data = TrainDataset(train_triplets, num_students, num_questions)

    # define model here
    model = IRTModel(**config)
    # train model
    model.init_model(train_data, pl=pl, num_dim=num_dim)  # 默认为3PL，8维

    if wandb_flag:
        model.train(train_data, log_step=10, wandb=wandb)
    else:
        model.train(train_data, log_step=10)

    wandb.finish()

    # save model
    model.adaptest_save(f'./save/{task}_irt_dim{num_dim}_{pl}pl.pt')

config = {
    'learning_rate': 0.002,
    'batch_size': 256,
    'num_epochs': 200,
    'device': 'cuda',
}

if __name__ == '__main__':
    # setting_info = [setting_info[i] for i in range(len(setting_info)) if setting_info[i]['pl'] == 1]

    # 多线程训练
    pool = Pool(len(setting_info))
    pool.map(run_train, range(len(setting_info)))
    pool.close()

    # 单线程训练
    # for i in range(len(setting_info)):
    #     print("dim: " + str(setting_info[i]['num_dim']) + " pl: " + str(setting_info[i]['pl']))
    #     run_train(i)
