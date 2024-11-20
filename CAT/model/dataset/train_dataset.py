import torch
from torch.utils import data

try:
    # for python module
    from .dataset import Dataset
except (ImportError, SystemError):  # pragma: no cover
    # for python script
    from dataset import Dataset


class TrainDataset(Dataset, data.dataset.Dataset):

    def __init__(self, data, num_students, num_questions):
        """
        Args:
            data: list, [(sid, qid, score)]
            num_students: int, total student number
            num_questions: int, total question number
        """
        super().__init__(data, num_students, num_questions)

    def __getitem__(self, item):
        sid, qid, score = self.raw_data[item]
        return sid, qid, score

    def __len__(self):
        return len(self.raw_data)
