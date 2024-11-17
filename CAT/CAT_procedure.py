import json
import logging
import pandas as pd
from CAT.utils.logger import setuplogger
from strategy.random_strategy import RandomStrategy
from strategy.MFI_strategy import MFIStrategy
from strategy.KLI_strategy import KLIStrategy

from model.dataset.adaptest_dataset import AdapTestDataset
from model.IRT import IRTModel

setuplogger()
# choose dataset here
dataset = 'resident_eval'
# choose concept_map here
concept_map = 'resident_concept_map'
# modify config here
config = {
    'learning_rate': 0.002,
    'batch_size': 64,
    'num_epochs': 200,
    'device': 'cuda',
    'num_dim': 1,  # for IRT or MIRT
}

# read datasets
test_triplets = pd.read_csv(f'./data/{dataset}.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'./concept_map/{concept_map}.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}

test_data = AdapTestDataset(test_triplets, concept_map,
                            9,
                            1836,
                            8)

# checkpoint path
ckpt_path_question = './save/irt.pt'
# fixed test length
test_length = 10
# choose strategies here
strategies = [
    # RandomStrategy(),
    MFIStrategy(),
    # KLIStrategy()
]

for strategy in strategies:
    # define model here
    model = IRTModel(**config)
    # load the parameters for questions
    model.init_model(test_data)
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
        logging.info("len of selected_questions: " + str(len(selected_questions)))
        logging.info("selected_questions: " + str(selected_questions))
        for student, question in selected_questions.items():
            test_data.apply_selection(student, question)
        # update model
        model.adaptest_update(test_data)
        # evaluate model
        results = model.evaluate(test_data)
        for name, value in results.items():
            logging.info(f'{name}:{value}')
