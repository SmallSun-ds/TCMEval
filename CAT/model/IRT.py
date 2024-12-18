import vegas
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import integrate

from .abstract_model import AbstractModel
from CAT.model.dataset.adaptest_dataset import AdapTestDataset
from CAT.model.dataset.train_dataset import TrainDataset
from CAT.model.dataset.dataset import Dataset

class IRT_1PL(nn.Module):
    def __init__(self, num_students, num_questions, num_dim):
        # num_dim: IRT if num_dim == 1 else MIRT
        super().__init__()
        self.num_dim = num_dim
        self.num_students = num_students
        self.num_questions = num_questions
        self.theta = nn.Embedding(self.num_students, self.num_dim)
        self.beta = nn.Embedding(self.num_questions, self.num_dim)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_ids, question_ids):
        theta = self.theta(student_ids)
        beta = self.beta(question_ids)
        # for IRT-1PL
        pred = (theta - beta).sum(dim=1, keepdim=True)
        pred = torch.sigmoid(pred)
        return pred

class IRT_2PL(nn.Module):
    def __init__(self, num_students, num_questions, num_dim):
        # num_dim: IRT if num_dim == 1 else MIRT
        super().__init__()
        self.num_dim = num_dim
        self.num_students = num_students
        self.num_questions = num_questions
        self.theta = nn.Embedding(self.num_students, self.num_dim)
        self.alpha = nn.Embedding(self.num_questions, self.num_dim)
        self.beta = nn.Embedding(self.num_questions, self.num_dim)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_ids, question_ids):
        theta = self.theta(student_ids)
        alpha = self.alpha(question_ids)
        beta = self.beta(question_ids)
        # for IRT-2PL
        pred = (alpha * (theta - beta)).sum(dim=1, keepdim=True)
        pred = torch.sigmoid(pred)
        return pred

class IRT_3PL(nn.Module):
    def __init__(self, num_students, num_questions, num_dim):
        # num_dim: IRT if num_dim == 1 else MIRT
        super().__init__()
        self.num_dim = num_dim
        self.num_students = num_students
        self.num_questions = num_questions
        self.theta = nn.Embedding(self.num_students, self.num_dim)  # (num_students, num_dim)
        self.alpha = nn.Embedding(self.num_questions, self.num_dim)  # (num_questions, num_dim)
        self.beta = nn.Embedding(self.num_questions, self.num_dim)  # (num_questions, num_dim)
        self.gamma_raw = nn.Embedding(self.num_questions, 1)  # gamma 的维度固定为 1

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_ids, question_ids):
        theta = self.theta(student_ids)  # (batch_size, num_dim)
        alpha = self.alpha(question_ids)  # (batch_size, num_dim)
        beta = self.beta(question_ids)  # (batch_size, num_dim)
        gamma = torch.sigmoid(self.gamma_raw(question_ids)).squeeze(-1)  # (batch_size,)

        # 计算预测值 (IRT-3PL)
        pred = (alpha * (theta - beta)).sum(dim=1)  # 按维度求和, 输出 (batch_size,)
        pred = torch.sigmoid(pred)  # (batch_size,)

        # IRT-3PL模型：添加猜测参数
        pred = gamma + (1 - gamma) * pred  # (batch_size,)
        return pred


class IRTModel(AbstractModel):

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.model = None

    @property
    def name(self):
        return 'Item Response Theory'

    def init_model(self, data: Dataset, pl=3, num_dim=8):
        if pl == 1:
            self.model = IRT_1PL(data.num_students, data.num_questions, num_dim)
        elif pl == 2:
            self.model = IRT_2PL(data.num_students, data.num_questions, num_dim)
        elif pl == 3:
            self.model = IRT_3PL(data.num_students, data.num_questions, num_dim)

    def train(self, train_data: TrainDataset, log_step=1, wandb=None):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        self.model.to(device)
        logging.info('train on {}'.format(device))

        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            loss = 0.0
            for cnt, (student_ids, question_ids, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                loss += bz_loss.data.float()
                if cnt % log_step == 0:
                    logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))
                    if wandb is not None:
                        wandb.log({
                            "epoch": ep,
                            "loss": loss / cnt,
                        })

    def adaptest_save_question(self, path):
        """
        Save the model. Only save the parameters of questions(alpha, beta)
        """
        model_dict = self.model.state_dict()
        model_dict = {k: v for k, v in model_dict.items() if 'alpha' in k or 'beta' in k}
        torch.save(model_dict, path)

    def adaptest_save_student(self, path):
        """
        Save the model. Only save the parameters of students(theta)
        """
        model_dict = self.model.state_dict()
        model_dict = {k: v for k, v in model_dict.items() if 'theta' in k}
        torch.save(model_dict, path)

    def adaptest_save(self, path):
        """
        Save the model. Only save the parameters of questions(alpha, beta)
        """
        model_dict = self.model.state_dict()
        model_dict = {k: v for k, v in model_dict.items() if 'alpha' in k or 'beta' in k}
        torch.save(model_dict, path)

    def adaptest_load(self, path):
        """
        Reload the saved model
        """
        print("load model from", path)
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])

    def adaptest_update(self, adaptest_data: AdapTestDataset):
        """
        Update CDM with tested data
        """
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)

        # last=True: only choose the last tested data as dataset
        # last=False: choose all tested data as dataset
        tested_dataset = adaptest_data.get_tested_dataset(last=False)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

        for ep in range(1, epochs + 1):
            loss = 0.0
            log_steps = 1
            for cnt, (student_ids, question_ids, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                loss += bz_loss.data.float()
                # if cnt % log_steps == 0:
                #     print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, cnt, loss / cnt))

    def evaluate(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        device = self.config['device']

        real = []
        pred = []
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                output = self.model(student_ids, question_ids).view(-1)
                pred += output.tolist()
            self.model.train()

        real = np.array(real)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)
        for i in range(len(pred)):
            if pred[i] < 0.5:
                pred[i] = 0
            else:
                pred[i] = 1
        acc = accuracy_score(real, pred)

        return {
            'acc': acc,
            'auc': auc,
        }

    def get_pred(self, adaptest_data: AdapTestDataset):
        """
        Returns:
            predictions, dict[sid][qid]
        """
        data = adaptest_data.data
        device = self.config['device']

        pred_all = {}

        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                output = self.model(student_ids, question_ids).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()

        return pred_all

    def _loss_function(self, pred, real):
        # print("pred.shape: ", pred.shape)
        # print("real.shape: ", real.shape)
        return -(real * torch.log(0.0001 + pred) + (1 - real) * torch.log(1.0001 - pred)).mean()

    # def _loss_function(self, pred, real):
    #     pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)  # 限制 pred 在 (1e-7, 1-1e-7) 范围内
    #     return -(real * torch.log(pred) + (1 - real) * torch.log(1 - pred)).mean()
    #
    # def _loss_function(self, pred, real):
    #     return nn.BCEWithLogitsLoss()(pred, real)

    def get_alpha(self, question_id):
        """ get alpha of one question
        Args:
            question_id: int, question id
        Returns:
            alpha of the given question, shape (num_dim, )
        """
        return self.model.alpha.weight.data.cpu().numpy()[question_id]

    def get_beta(self, question_id):
        """ get beta of one question
        Args:
            question_id: int, question id
        Returns:
            beta of the given question, shape (1, )
        """
        return self.model.beta.weight.data.cpu().numpy()[question_id]

    def get_theta(self, student_id):
        """ get theta of one student
        Args:
            student_id: int, student id
        Returns:
            theta of the given student, shape (num_dim, )
        """
        return self.model.theta.weight.data.cpu().numpy()[student_id]

    def get_gamma(self, question_id):
        """ get gamma of one question
        Args:
            question_id: int, question id
        Returns:
            gamma of the given question, shape (1, )
        """

        return torch.sigmoid(self.model.gamma_raw.weight.data[question_id]).cpu().numpy()

    def get_kli(self, student_id, question_id, n, pred_all):
        """ get KL information
        Args:
            student_id: int, student id
            question_id: int, question id
            n: int, the number of iteration
        Returns:
            v: float, KL information
        """
        if n == 0:
            return np.inf
        device = self.config['device']
        dim = self.model.num_dim
        sid = torch.LongTensor([student_id]).to(device)
        qid = torch.LongTensor([question_id]).to(device)
        theta = self.get_theta(sid)  # (num_dim, )
        alpha = self.get_alpha(qid)  # (num_dim, )
        beta = self.get_beta(qid)[0]  # float value
        pred_estimate = pred_all[student_id][question_id]

        def kli(x):
            """ The formula of KL information. Used for integral.
            Args:
                x: theta of student sid
            """
            if type(x) == float:
                x = np.array([x])
            # for IRT-2PL
            pred = np.matmul(alpha.T, (x - beta))
            # for IRT-1PL
            # pred = x - beta
            pred = 1 / (1 + np.exp(-pred))
            q_estimate = 1 - pred_estimate
            q = 1 - pred
            return pred_estimate * np.log(pred_estimate / pred) + \
                q_estimate * np.log((q_estimate / q))

        c = 3
        boundaries = [[theta[i] - c / np.sqrt(n), theta[i] + c / np.sqrt(n)] for i in range(dim)]
        if len(boundaries) == 1:
            # KLI
            v, err = integrate.quad(kli, boundaries[0][0], boundaries[0][1])
            return v
        # MKLI
        integ = vegas.Integrator(boundaries)
        result = integ(kli, nitn=10, neval=1000)
        return result.mean

    def get_fisher(self, student_id, question_id, pred_all):
        """ get Fisher information
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            fisher_info: matrix(num_dim * num_dim), Fisher information
        """
        device = self.config['device']
        qid = torch.LongTensor([question_id]).to(device)
        pred = pred_all[student_id][question_id]
        q = 1 - pred
        if isinstance(self.model, IRT_2PL):  # 2PL
            alpha = self.model.alpha(qid).clone().detach().cpu()
            fisher_info = (q * pred * (alpha * alpha.T)).numpy()
        elif isinstance(self.model, IRT_1PL):  # 1PL
            fisher_info = (q * pred)
        else:  # 3PL
            alpha = self.model.alpha(qid).clone().detach().cpu()
            gamma = torch.sigmoid(self.model.gamma_raw(qid)).squeeze(-1).clone().detach().cpu()
            # I(\theta) = \frac{\alpha^2 \cdot (p(\theta) - \gamma)^2 \cdot (1 - p(\theta))}{(1 - \gamma)^2 \cdot p(\theta)}
            fisher_info = (alpha * alpha.T * (pred - gamma) * (pred - gamma) * q / (1 - gamma) / (1 - gamma) / pred).numpy()
        # print("fisher_info.shape", fisher_info.shape)
        return fisher_info


    def expected_model_change(self, sid: int, qid: int, adaptest_data: AdapTestDataset, pred_all: dict):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        epochs = self.config['num_epochs']
        lr = self.config['learning_rate']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for name, param in self.model.named_parameters():
            if 'theta' not in name:
                param.requires_grad = False

        original_weights = self.model.theta.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        correct = torch.LongTensor([1]).to(device).float()
        wrong = torch.LongTensor([0]).to(device).float()

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
            (1 - pred) * torch.norm(neg_weights - original_weights).item()
