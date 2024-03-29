# import numpy as np
from sklearn.model_selection import train_test_split


class DatasetHandler:

    def __init__(self, encoder_tokenizer, decoder_tokenizer, split=0.05):
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

    def get_train(self, num_instances=10000):
        return self.source_train[:num_instances], self.target_train[:num_instances, :-1], self.target_train[:num_instances, 1:]

    def get_test(self, num_instances=100):
        return self.source_test[:num_instances], self.target_test[:num_instances, :-1], self.target_test[:num_instances, 1:]

