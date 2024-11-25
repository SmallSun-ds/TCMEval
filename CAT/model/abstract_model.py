from abc import ABC, abstractmethod
from CAT.model.dataset.adaptest_dataset import AdapTestDataset
from CAT.model.dataset.train_dataset import TrainDataset
from CAT.model.dataset.dataset import Dataset


class AbstractModel(ABC):

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def adaptest_update(self, adaptest_data: AdapTestDataset):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, adaptest_data: AdapTestDataset):
        raise NotImplementedError

    @abstractmethod
    def init_model(self, data: Dataset):
        raise NotImplementedError

    @abstractmethod
    def train(self, train_data: TrainDataset):
        raise NotImplementedError

    @abstractmethod
    def adaptest_save(self, path):
        raise NotImplementedError

    @abstractmethod
    def adaptest_load(self, path):
        raise NotImplementedError