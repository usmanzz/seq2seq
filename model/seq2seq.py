from __future__ import print_function
from tensorflow.python.keras.layers import Input, Dense, CuDNNLSTM, TimeDistributed, Concatenate, LSTM, Embedding
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.keras.models import Model
import numpy as np
from abc import ABCMeta, abstractmethod
from ..layers.attention import AttentionLayer


class EncoderDecoder(metaclass=ABCMeta):
    def __init__(self, latent_dim, data, embedding_dim=100):
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.data = data
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.combined = None

    def train(self, num_instances=10000, batch_size=100, epochs=100):
        source, inp_targets, targets = self.data.get_train(num_instances)
        self.combined.fit([source, inp_targets], targets,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_split=0.1)

    def test(self, num_instances=100):
        source, target, = self.data.get_test(num_instances)
        for ind, input_seq in enumerate(source):
            decoded_sentence = self.decode_seq(input_seq)
            print('-')
            print('Input sentence:', self.data.encoder_tokenizer.nums2seq(source[ind]))
            print('targrt sentence:', self.data.decoder_tokenizer.nums2seq(target[ind]))
            print('Decoded sentence:', decoded_sentence)

    def save(self, name="s2s"):
        self.combined.save(name + '.h5')

    def load(self, name="s2s"):
        self.combined.load(name + '.h5')

    def get_encoder(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        embeddings = Embedding(self.data.num_encoder_tokens, self.embedding_dim)(encoder_inputs)
        # encoder = CuDNNLSTM(self.latent_dim, return_state=True, return_sequences=True)
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs = encoder(embeddings)
        encoder_model = Model(encoder_inputs, encoder_outputs)
        encoder_model.summary()
        return encoder_model

    @abstractmethod
    def get_decoder(self):
        pass

    @abstractmethod
    def decode_seq(self, input_seq):
        pass


class Seq2seq(EncoderDecoder):
    def __init__(self, latent_dim, data, embedding_dim=100):
        super().__init__(latent_dim, data, embedding_dim)
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = Input(shape=(None,))
        _, e_state_h, e_state_c = self.encoder(encoder_inputs)
        decoded_output, state_h, state_c = self.decoder([decoder_inputs, e_state_h, e_state_c])
        self.combined = Model([encoder_inputs, decoder_inputs], decoded_output)
        self.combined.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        # self.auto_encoder.summary()

    def get_decoder(self):
        decoder_inputs = Input(shape=(None,))
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        embeddings = Embedding(self.data.num_decoder_tokens, self.embedding_dim)(decoder_inputs)
        # decoder_lstm = CuDNNLSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, state_h, state_c = decoder_lstm(embeddings,
                                                         initial_state=[decoder_state_input_h, decoder_state_input_c])
        decoder_dense = Dense(self.data.num_decoder_tokens, activation='softmax')
        decoded_output = TimeDistributed(decoder_dense)(decoder_outputs)
        decoder_model = Model([decoder_inputs, decoder_state_input_h, decoder_state_input_c],
                              [decoded_output, state_h, state_c])
        decoder_model.summary()
        return decoder_model

    def decode_seq(self, input_seq):
        e_out, dsh, dsc = self.encoder.predict(input_seq.reshape((1, -1)))
        # Generate empty target sequence of length 1.
        target_seq = np.array([[self.data.decoder_tokenizer.start_tkn]])
        # Sampling loop for a batch of sequences
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, dsh, dsc = self.decoder.predict([target_seq, dsh, dsc])
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            decoded_sentence.append(sampled_token_index)
            # Exit condition: either hit max length or find stop character.
            if (sampled_token_index == self.data.decoder_tokenizer.end or len(
                    decoded_sentence) > self.data.max_decoder_seq_length):
                stop_condition = True
            target_seq = np.array([[sampled_token_index]])
        return self.data.decoder_tokenizer.nums2seq(decoded_sentence)


class Seq2seqAttention(EncoderDecoder):

    def __init__(self, latent_dim, data, embedding_dim=100):
        super().__init__(latent_dim, data, embedding_dim)
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = Input(shape=(None,))
        encoder_outputs, e_state_h, e_state_c = self.encoder(encoder_inputs)
        decoded_output, state_h, state_c = self.decoder([decoder_inputs, encoder_outputs, e_state_h, e_state_c])
        self.combined = Model([encoder_inputs, decoder_inputs], decoded_output)
        self.combined.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

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
        e_out, dsh, dsc = self.encoder.predict(input_seq.reshape((1, -1)))
        # Generate empty target sequence of length 1.
        target_seq = np.array([[self.data.decoder_tokenizer.start_tkn]])
        # Sampling loop for a batch of sequences
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, dsh, dsc = self.decoder.predict([target_seq, e_out, dsh, dsc])
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            decoded_sentence.append(sampled_token_index)
            # Exit condition: either hit max length or find stop character.
            if (sampled_token_index == self.data.decoder_tokenizer.end or len(
                    decoded_sentence) > self.data.max_decoder_seq_length):
                stop_condition = True
            target_seq = np.array([[sampled_token_index]])
        return self.data.decoder_tokenizer.nums2seq(decoded_sentence)


class BiSeq2seqAttention(EncoderDecoder):

    def __init__(self, latent_dim, data, embedding_dim=100):
        super().__init__(latent_dim, data, embedding_dim)
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = Input(shape=(None,))
        encoder_outputs = self.encoder(encoder_inputs)
        decoded_output = self.decoder([decoder_inputs]+encoder_outputs)
        self.combined = Model([encoder_inputs, decoder_inputs], decoded_output[0])
        self.combined.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        # self.auto_encoder.summary()

    def get_encoder(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        embeddings = Embedding(self.data.num_encoder_tokens, self.embedding_dim)(encoder_inputs)
        # encoder = CuDNNLSTM(self.latent_dim, return_state=True, return_sequences=True)
        encoder = LSTM(self.latent_dim, return_state=True, return_sequences=True)
        encoder_outputs = Bidirectional(encoder)(embeddings)
        encoder_model = Model(encoder_inputs, encoder_outputs)
        encoder_model.summary()
        return encoder_model

    def get_decoder(self):
        decoder_inputs = Input(shape=(None,))
        dsh = Input(shape=(self.latent_dim,))
        dsc = Input(shape=(self.latent_dim,))
        dsh1 = Input(shape=(self.latent_dim,))
        dsc1 = Input(shape=(self.latent_dim,))
        encoder_outputs = Input(shape=(self.data.max_encoder_seq_length, self.latent_dim))
        embeddings = Embedding(self.data.num_decoder_tokens, self.embedding_dim)(decoder_inputs)
        # decoder_lstm = CuDNNLSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        outputs = Bidirectional(decoder_lstm)(embeddings, initial_state=[dsh, dsc, dsh1, dsc1])
        decoder_outputs = outputs[0]
        attn_layer = AttentionLayer(name='attention_layer')
        # print(encoder_outputs.shape, decoder_outputs.shape)
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        context_vectors = Concatenate()([attn_out, decoder_outputs])
        decoder_dense = Dense(self.data.num_decoder_tokens, activation='softmax')
        decoded_output = TimeDistributed(decoder_dense)(context_vectors)
        decoder_model = Model([decoder_inputs, encoder_outputs, dsh, dsc, dsh1, dsc1],
                              [decoded_output]+outputs[1:])
        decoder_model.summary()
        return decoder_model

    def decode_seq(self, input_seq):
        e_out = self.encoder.predict(input_seq.reshape((1, -1)))
        # Generate empty target sequence of length 1.
        target_seq = np.array([[self.data.decoder_tokenizer.start_tkn]])
        # Sampling loop for a batch of sequences
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, dsh, dsc = self.decoder.predict([target_seq]+e_out)
            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            decoded_sentence.append(sampled_token_index)
            # Exit condition: either hit max length or find stop character.
            if (sampled_token_index == self.data.decoder_tokenizer.end or len(
                    decoded_sentence) > self.data.max_decoder_seq_length):
                stop_condition = True
            target_seq = np.array([[sampled_token_index]])
        return self.data.decoder_tokenizer.nums2seq(decoded_sentence)
