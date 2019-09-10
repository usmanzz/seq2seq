from model.seq2seq import Seq2seq
from utils.data import Eng2Fra, Fra2Eng

# import numpy as np

if __name__ == '__main__':

    data = Eng2Fra()
    seq2seq = Seq2seq(100, data)
