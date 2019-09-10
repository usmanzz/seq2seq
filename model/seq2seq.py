from __future__ import print_function
from tensorflow.python.keras.layers import Input, Dense, CuDNNLSTM, TimeDistributed, Concatenate, LSTM, Embedding
from tensorflow.python.keras.models import Model
import numpy as np

from ..layers.attention import AttentionLayer
from ..utils.data import Eng2Fra


class Seq2seq:

    def __init__(self, latent_dim, data, embedding_dim=100):
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.data = data
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = Input(shape=(None,))

        encoder_outputs, e_state_h, e_state_c = self.encoder(encoder_inputs)
        print("encoder : ", encoder_inputs.shape, encoder_outputs.shape)
        decoded_output, state_h, state_c = self.decoder([decoder_inputs, encoder_outputs, e_state_h, e_state_c])
        self.auto_encoder = Model([encoder_inputs, decoder_inputs], decoded_output)
        self.auto_encoder.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        # self.auto_encoder.summary()

    def train(self, source, inp_targets, targets, batch_size=100, epochs=100):
        self.auto_encoder.fit([source, inp_targets], targets,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.2)

    def test(self, source, target):
        for ind, input_seq in enumerate(source):
            decoded_sentence = self.decode_seq(input_seq)
            print('-')
            print('Input sentence:', self.data.encoder_tokenizer.nums2seq(target[ind]))
            print('targrt sentence:', self.data.decoder_tokenizer.nums2seq(input_seq))
            print('Decoded sentence:', decoded_sentence)

    def save(self, name="s2s"):
        self.auto_encoder.save(name + '.h5')

    def load(self, name="s2s"):
        self.auto_encoder.load(name + '.h5')

    def get_encoder(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        embeddings = Embedding(self.data.num_encoder_tokens, self.embedding_dim)(encoder_inputs)
        # encoder = CuDNNLSTM(self.latent_dim, return_state=True, return_sequences=True)
        encoder = LSTM(self.latent_dim, return_state=True, return_sequences=True)
        encoder_outputs = encoder(embeddings)
        encoder_model = Model(encoder_inputs, encoder_outputs)
        encoder_model.summary()
        return encoder_model

    def get_decoder(self):
        decoder_inputs = Input(shape=(None,))
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        encoder_outputs = Input(shape=(self.data.max_encoder_seq_length, self.latent_dim))
        embeddings = Embedding(self.data.num_decoder_tokens, self.embedding_dim)(decoder_inputs)
        # decoder_lstm = CuDNNLSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, state_h, state_c = decoder_lstm(embeddings,
                                                         initial_state=[decoder_state_input_h, decoder_state_input_c])
        attn_layer = AttentionLayer(name='attention_layer')
        # print(encoder_outputs.shape, decoder_outputs.shape)
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        context_vectors = Concatenate()([attn_out, decoder_outputs])
        decoder_dense = Dense(self.data.num_decoder_tokens, activation='softmax')
        decoded_output = TimeDistributed(decoder_dense)(context_vectors)
        decoder_model = Model([decoder_inputs, encoder_outputs, decoder_state_input_h, decoder_state_input_c],
                              [decoded_output, state_h, state_c])
        decoder_model.summary()
        return decoder_model

    def decode_seq(self, input_seq):
        e_out, dsh, dsc = self.encoder.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = self.data.decoder_tokenizer.start_tkn
        #     print(target_seq.shape)
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, dsh, dsc = self.decoder.predict([target_seq, e_out, dsh, dsc])
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            decoded_sentence.append(sampled_token_index)
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_token_index == self.data.decoder_tokenizer.end or
                    len(decoded_sentence) > self.data.max_decoder_seq_length):
                stop_condition = True
            target_seq = np.array([[sampled_token_index]])
        return self.data.decoder_tokenizer.nums2seq(decoded_sentence)
