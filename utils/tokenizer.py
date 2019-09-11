# import numpy as np
from keras_preprocessing.sequence import pad_sequences
from abc import ABCMeta, abstractmethod

import spacy, os

os.system('python -c "import thinc.neural.gpu_ops"')


class SeqTokenizer(metaclass=ABCMeta):

    def __init__(self, examples, is_encoder):
        self.examples = examples
        self.is_encoder = is_encoder
        self.start = "<s>"
        self.end = "<e>"
        self.start_tkn = 1
        self.end_tkn = 2
        self.vocab = {self.start: self.start_tkn, self.end: self.end_tkn}
        self.reverse_index = {}
        self._tokens = []
        self.seqs = []
        self.max_seq_len = 0
        self.num_tokens = 0
        self.process_data()

    """
        Tokenize each instance
        """

    @abstractmethod
    def tokenize_seq(self, seq):
        return seq

    def process_data(self):
        processed = [self.add_seq(self.tokenize_seq(seq)) for seq in self.examples]
        bt = len(self.vocab.keys())
        self.vocab.update({word: ind + bt for ind, word in enumerate(sorted(list(set(self._tokens))))})
        self.num_tokens = len(self.vocab.keys())
        self.reverse_index = {ind: word for word, ind in self.vocab.items()}
        self.seqs = pad_sequences([self.seq2nums(seq) for seq in processed], maxlen=self.max_seq_len, truncating="post",
                                  padding="post")

    def add_seq(self, seq_tokens):
        self._tokens.extend(seq_tokens)
        seq_length = len(seq_tokens) + 2 if self.is_encoder else 0
        if seq_length > self.max_seq_len:
            self.max_seq_len = seq_length
        return seq_tokens

    def seq2nums(self, sequence):
        nums = [self.vocab[word] for word in sequence]
        return nums if self.is_encoder else self.add_boarder_token(nums)

    def nums2seq(self, nums):
        return [self.reverse_index[num] for num in nums if num != 0]

    """implement this if you want different view"""
    def view_data(self, nums):
        print(" ".join(self.nums2seq(nums)))

    def get_batch(self, indices):
        self.seqs[indices]

    def add_boarder_token(self, seq):
        return [self.start_tkn] + seq + [self.end_tkn]


class EngTokenizer(SeqTokenizer):

    def __init__(self, examples, is_encoder=False):
        self.eng = spacy.load('en')
        super().__init__(examples, is_encoder)

    def tokenize_seq(self, seq):
        return [str(word) for word in self.eng(seq)]


class FraTokenizer(SeqTokenizer):

    def __init__(self, examples, is_encoder=False):
        self.fra = spacy.load('fr')
        super().__init__(examples, is_encoder)

    def tokenize_seq(self, seq):
        return [str(word) for word in self.fra(seq)]
