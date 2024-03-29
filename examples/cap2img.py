from ..utils.tokenizer import SeqTokenizer
from ..utils.data import DatasetHandler
# from model.seq2seq import Seq2seq
from .nmt import EngTokenizer
import matplotlib.pyplot as plt


class ImgTokenizer(SeqTokenizer):

    def __init__(self, img_shape, examples, is_encoder=False):
        self.img_shape = img_shape
        super().__init__(examples, is_encoder)

    def tokenize_seq(self, seq):
        return seq

    def view_data(self, data):
        n = len(data[0])
        plt.figure(figsize=(n * 2, 4))
        for ind, item in enumerate(list(zip(*data))):
            ax = plt.subplot(2, n, ind + 1)
            plt.imshow(item[0].reshape(self.img_shape))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, ind + 1 + n)
            plt.imshow(item[1].reshape(self.img_shape))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


class Cap2Img(DatasetHandler):

    def __init__(self, english_sentences, img_sequences, img_shape):
        encoder_tokenizer = EngTokenizer(english_sentences, is_encoder=True)
        decoder_tokenizer = ImgTokenizer(img_shape, img_sequences)
        super().__init__(encoder_tokenizer, decoder_tokenizer)


class Img2Cap(DatasetHandler):

    def __init__(self, english_sentences, img_sequences, img_shape):
        encoder_tokenizer = ImgTokenizer(img_shape, img_sequences, is_encoder=True)
        decoder_tokenizer = EngTokenizer(english_sentences)
        super().__init__(encoder_tokenizer, decoder_tokenizer)
