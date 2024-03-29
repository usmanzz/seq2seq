from ..utils.tokenizer import SeqTokenizer
from ..utils.data import DatasetHandler
import spacy, os
from ..model.seq2seq import Seq2seqAttention

os.system('python -c "import thinc.neural.gpu_ops"')


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


class Eng2Fra(DatasetHandler):

    def __init__(self, english_sentences, french_sentences, num_instances=10000):
        encoder_tokenizer = EngTokenizer(english_sentences[:num_instances], is_encoder=True)
        decoder_tokenizer = FraTokenizer(french_sentences[:num_instances])
        super().__init__(encoder_tokenizer, decoder_tokenizer)


class Fra2Eng(DatasetHandler):

    def __init__(self, english_sentences, french_sentences, num_instances=10000):
        encoder_tokenizer = FraTokenizer(french_sentences[:num_instances], is_encoder=True)
        decoder_tokenizer = EngTokenizer(english_sentences[:num_instances])
        super().__init__(encoder_tokenizer, decoder_tokenizer)


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv("fra.txt", sep='\t', names=["eng", "fra"])
    df.tail()
    data = Fra2Eng(df["eng"].values, df["fra"].values)
    seq2seq = Seq2seqAttention(100, data)
    seq2seq.train()
    seq2seq.test()