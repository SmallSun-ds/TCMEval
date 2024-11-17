import numpy as np

from .abstract_strategy import AbstractStrategy
from model.abstract_model import AbstractModel
from model.dataset.adaptest_dataset import AdapTestDataset


class RandomStrategy(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Random Select Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            selection[sid] = untested_questions[np.random.randint(len(untested_questions))]
        return selection