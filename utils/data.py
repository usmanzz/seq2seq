# import numpy as np
from utils.tokenizer import EngTokenizer, FraTokenizer
from sklearn.model_selection import train_test_split


class DatasetHandler:

    def __init__(self, encoder_tokenizer, decoder_tokenizer, split=0.9):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.num_decoder_tokens = self.decoder_tokenizer.num_tokens
        self.max_decoder_seq_length = self.decoder_tokenizer.max_seq_len
        self.num_encoder_tokens = self.encoder_tokenizer.num_tokens
        self.max_encoder_seq_length = self.encoder_tokenizer.max_seq_len
        self.source_train, self.source_test, self.target_train, self.target_test = train_test_split(
            self.encoder_tokenizer.seqs,
            self.decoder_tokenizer.seqs,
            test_size=split,
            random_state=42)

    def get_train(self):
        return self.source_train, self.target_train[:, :-1], self.target_train[:, 1:]

    def get_test(self):
        return self.source_test, self.target_test[:, :-1], self.target_test[:, 1:]


class Eng2Fra(DatasetHandler):

    def __init__(self, english_sentences, french_sentences):
        encoder_tokenizer = EngTokenizer(english_sentences)
        decoder_tokenizer = FraTokenizer(french_sentences)
        super().__init__(encoder_tokenizer, decoder_tokenizer)


class Fra2Eng(DatasetHandler):

    def __init__(self, english_sentences, french_sentences):
        encoder_tokenizer = FraTokenizer(french_sentences)
        decoder_tokenizer = EngTokenizer(english_sentences)
        super().__init__(encoder_tokenizer, decoder_tokenizer)
