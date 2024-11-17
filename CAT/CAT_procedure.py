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

index = 0
# {"task": "resident", "dataset": "resident_test", "concept_map": "resident_concept_map",
#  "num_students": 1, "num_questions": 1836, "num_concepts": 8},
# {"task": "resident", "dataset": "resident_train", "concept_map": "resident_concept_map",
#  "num_students": 10, "num_questions": 1836, "num_concepts": 8},
setting_info = [
    {"task": "resident", "dataset": "resident_eval", "concept_map": "resident_concept_map",
     "num_students": 9, "num_questions": 1836, "num_concepts": 8},
]

config = {
    'learning_rate': 0.002,
    'batch_size': 64,
    'num_epochs': 200,
    'device': 'cuda',
    'num_dim': 1,  # IRT if num_dim == 1 else MIRT
}

# fixed test length（选取题目的个数）
test_length = 50

task = setting_info[index]["task"]
dataset = setting_info[index]["dataset"]
concept_map = setting_info[index]["concept_map"]
num_students = setting_info[index]["num_students"]
num_questions = setting_info[index]["num_questions"]
num_concepts = setting_info[index]["num_concepts"]

# read datasets
test_triplets = pd.read_csv(f'./data/{dataset}.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'./concept_map/{concept_map}.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}

test_data = AdapTestDataset(test_triplets, concept_map,
                            num_students,  # 同时为num_students个学生进行CAT测试
                            num_questions,
                            num_concepts)

# checkpoint path
ckpt_path_question = f'./save/{task}_irt.pt'
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
        # selected_questions_list.append(selected_questions)
        for selected_question in selected_questions.items():
            correct = test_data.get_score(selected_question[0],
                                          selected_question[1])  # 实际上需要运行本地LLM（需要GPU）或者消耗API，但是已有label（corrcet），所以直接获取
            # logging.info(f'student {selected_question[0]} select question {selected_question[1]} with score {corrcet}')
            examinees_scores.append({
                'student': selected_question[0],
                'question': selected_question[1],
                'score': correct  # 0 or 1，0为错误，1为正确
            })
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

student_ids = set(x['student'] for x in examinees_scores)
# 计算每个学生的准确率
student_scores_CAT = {}
logging.info("CAT score:")
for student_id in student_ids:
    student_scores_CAT[student_id] = sum(
        [x['score'] for x in examinees_scores if x['student'] == student_id]) / test_length
    logging.info("student_ids: " + str(student_id) + " score: " + str(student_scores_CAT[student_id]))

# 计算总体准确率(使用test_triplets)
student_scores_total = {}
logging.info("Total score:")
for student_id in student_ids:
    student_scores_total[student_id] = sum([x[2] for x in test_triplets if x[0] == student_id]) / len(
        [x[2] for x in test_triplets if x[0] == student_id])
    # print(len([x[2] for x in test_triplets if x[0] == student_id]))
    logging.info("student_ids: " + str(student_id) + " score: " + str(student_scores_total[student_id]))

# 计算皮尔逊相关系数、Spearman相关系数、Kendall相关系数（student_scores_CAT和student_scores_total）
from scipy.stats import pearsonr, spearmanr, kendalltau

student_scores_CAT_list = [student_scores_CAT[x] for x in student_scores_CAT.keys()]
student_scores_total_list = [student_scores_total[x] for x in student_scores_total.keys()]
# print("student_scores_CAT_list: " + str(student_scores_CAT_list))
# print("student_scores_total_list: " + str(student_scores_total_list))
pearson_corr, _ = pearsonr(student_scores_CAT_list, student_scores_total_list)
spearman_corr, _ = spearmanr(student_scores_CAT_list, student_scores_total_list)
kendall_corr, _ = kendalltau(student_scores_CAT_list, student_scores_total_list)
logging.info("Pearson correlation coefficient: " + str(pearson_corr))
logging.info("Spearman correlation coefficient: " + str(spearman_corr))
logging.info("Kendall correlation coefficient: " + str(kendall_corr))
