from collections import defaultdict, deque


class Dataset(object):

    def __init__(self, data,
                 num_students, num_questions):
        """
        Args:
            data: list, [(sid, qid, score)]
            num_students: int, total student number
            num_questions: int, total question number
        """
        self._raw_data = data
        self.n_students = num_students
        self.n_questions = num_questions
        
        # reorganize datasets
        self._data = {}
        for sid, qid, correct in data:
            self._data.setdefault(sid, {})
            self._data[sid].setdefault(qid, {})
            self._data[sid][qid] = correct

        student_ids = set(x[0] for x in data)
        question_ids = set(x[1] for x in data)

        # print("student_ids: " + str(student_ids))
        # print("num_students: " + str(num_students))
        # print("num_questions: " + str(num_questions))

        assert max(student_ids) < num_students, \
            'Require student ids renumbered'
        assert max(question_ids) < num_questions, \
            'Require student ids renumbered'

    @property
    def num_students(self):
        return self.n_students

    @property
    def num_questions(self):
        return self.n_questions

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data
