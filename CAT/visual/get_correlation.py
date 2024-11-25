import json
from CAT.utils.logger import setuplogger
import pandas as pd
from CAT.utils.settings import setting_info

setuplogger()


def run_get_correlation(index):
    task = setting_info[index]["task"]
    dataset = setting_info[index]["dataset"]
    num_dim = setting_info[index]['num_dim']
    pl = setting_info[index]['pl']

    # 读取数据集
    with open(f'../result/{task}_dim{num_dim}_{pl}pl.json', 'r') as f:
        examinees_scores = json.load(f)
    test_triplets = pd.read_csv(f'../data/{dataset}.csv', encoding='utf-8').to_records(index=False)
    test_length = 50
    student_ids = set(x['student'] for x in examinees_scores)
    # 计算每个学生的准确率
    student_scores_CAT = {}
    # logging.info("CAT score:")
    for student_id in student_ids:
        student_scores_CAT[student_id] = sum(
            [x['score'] for x in examinees_scores if x['student'] == student_id]) / test_length
        # logging.info("student_ids: " + str(student_id) + " score: " + str(student_scores_CAT[student_id]))

    # 计算总体准确率(使用test_triplets)
    student_scores_total = {}
    # logging.info("Total score:")
    for student_id in student_ids:
        student_scores_total[student_id] = sum([x[2] for x in test_triplets if x[0] == student_id]) / len(
            [x[2] for x in test_triplets if x[0] == student_id])
        # print(len([x[2] for x in test_triplets if x[0] == student_id]))
        # logging.info("student_ids: " + str(student_id) + " score: " + str(student_scores_total[student_id]))

    # 计算皮尔逊相关系数、Spearman相关系数、Kendall相关系数（student_scores_CAT和student_scores_total）
    from scipy.stats import pearsonr, spearmanr, kendalltau

    student_scores_CAT_list = [student_scores_CAT[x] for x in student_scores_CAT.keys()]
    student_scores_total_list = [student_scores_total[x] for x in student_scores_total.keys()]
    # print("student_scores_CAT_list: " + str(student_scores_CAT_list))
    # print("student_scores_total_list: " + str(student_scores_total_list))
    pearson_corr, _ = pearsonr(student_scores_CAT_list, student_scores_total_list)
    spearman_corr, _ = spearmanr(student_scores_CAT_list, student_scores_total_list)
    kendall_corr, _ = kendalltau(student_scores_CAT_list, student_scores_total_list)
    # logging.info("Pearson correlation coefficient: " + str(pearson_corr))
    # logging.info("Spearman correlation coefficient: " + str(spearman_corr))
    # logging.info("Kendall correlation coefficient: " + str(kendall_corr))

    return pearson_corr, spearman_corr, kendall_corr


if __name__ == '__main__':
    # setting_info = [setting_info[i] for i in range(1, len(setting_info), 3)]
    for i in range(len(setting_info)):
        print("\nnum_dim: " + str(setting_info[i]['num_dim']) + " pl: " + str(setting_info[i]['pl']))
        pearson_corr, spearman_corr, kendall_corr = run_get_correlation(i)
        # logging.info("Pearson correlation coefficient: " + str(pearson_corr))
        # logging.info("Spearman correlation coefficient: " + str(spearman_corr))
        # logging.info("Kendall correlation coefficient: " + str(kendall_corr))
        print("Pearson correlation coefficient: " + str(pearson_corr))
        print("Spearman correlation coefficient: " + str(spearman_corr))
        print("Kendall correlation coefficient: " + str(kendall_corr))
