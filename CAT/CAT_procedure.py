import json
import logging
import pandas as pd
from CAT.utils.logger import setuplogger
from strategy.random_strategy import RandomStrategy
from strategy.MFI_strategy import MFIStrategy
from strategy.KLI_strategy import KLIStrategy
from model.dataset.adaptest_dataset import AdapTestDataset
from model.IRT import IRTModel
from multiprocessing import Pool
from CAT.utils.settings import test_setting_info as setting_info
from CAT.utils.settings import test_length

setuplogger()


def run_test(index):
    task = setting_info[index]["task"]
    dataset = setting_info[index]["dataset"]
    num_students = setting_info[index]["num_students"]
    num_questions = setting_info[index]["num_questions"]
    num_dim = setting_info[index]['num_dim']
    pl = setting_info[index]['pl']

    # read datasets
    test_triplets = pd.read_csv(f'./data/{dataset}.csv', encoding='utf-8').to_records(index=False)

    test_data = AdapTestDataset(test_triplets,
                                num_students,  # 同时为num_students个学生进行CAT测试
                                num_questions)

    # checkpoint path
    ckpt_path_question = f'./save/{task}_irt_dim{num_dim}_{pl}pl.pt'
    # choose strategies here
    strategies = [
        # RandomStrategy(),
        MFIStrategy(),
        # KLIStrategy()
    ]

    examinees_scores = []
    for strategy in strategies:
        # define model here
        model = IRTModel(**config)
        # load the parameters for questions
        model.init_model(test_data, pl=pl, num_dim=num_dim)
        model.adaptest_load(ckpt_path_question)

        test_data.reset()
        logging.info(f'start adaptive testing with {strategy.name} strategy')
        logging.info(f'Iteration 0')
        # evaluate model
        results = model.evaluate(test_data)
        for name, value in results.items():
            logging.info(f'{name}:{value}')

        for it in range(1, test_length + 1):
            logging.info(f'Iteration {it}')
            # select question
            selected_questions = strategy.adaptest_select(model, test_data)
            for selected_question in selected_questions.items():
                correct = test_data.get_score(selected_question[0],
                                              selected_question[1])  # 实际上需要运行本地LLM（需要GPU）或者消耗API，但是已有label（corrcet），所以直接获取
                examinees_scores.append({
                    'student': int(selected_question[0]),
                    'question': int(selected_question[1]),
                    'score': int(correct)  # 0 or 1，0为错误，1为正确
                })
            # logging.info("len of selected_questions: " + str(len(selected_questions)))
            # logging.info("selected_questions: " + str(selected_questions))
            for student, question in selected_questions.items():
                test_data.apply_selection(student, question)
            # update model
            model.adaptest_update(test_data)
            # evaluate model
            results = model.evaluate(test_data)
            for name, value in results.items():
                logging.info(f'{name}:{value}')
    # 保存examinees_scores
    with open(f'./result/{task}_dim{num_dim}_{pl}pl.json', 'w') as f:
        json.dump(examinees_scores, f)

config = {
    'learning_rate': 0.002,
    'batch_size': 64,
    'num_epochs': 200,
    'device': 'cuda',
}

if __name__ == '__main__':
    # setting_info = [setting_info[i] for i in range(len(setting_info)) if setting_info[i]['pl'] == 1]

    # 多线程运行
    pool = Pool(len(setting_info))
    pool.map(run_test, range(len(setting_info)))
    pool.close()

    # 单线程运行
    # for i in range(len(setting_info)):
    #     print("dim: " + str(setting_info[i]['num_dim']) + " pl: " + str(setting_info[i]['pl']))
    #     run_test(i)
