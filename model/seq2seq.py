from __future__ import print_function
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Concatenate, LSTM, Embedding
from tensorflow.keras.layers import Bidirectional, CuDNNLSTM
from tensorflow.keras.models import Model
import numpy as np
from abc import ABCMeta, abstractmethod
import tensorflow.keras.backend as K
from ..layers.attention import AttentionLayer


class EncoderDecoder(metaclass=ABCMeta):
    def __init__(self, latent_dim, data, embedding_dim=100):
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.data = data
        self.encoder_layer = LSTM(self.latent_dim, return_state=True, return_sequences=True)
        self.decoder_layer = LSTM(self.latent_dim, return_sequences=True, return_state=True)
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
        source, target, _ = self.data.get_test(num_instances)
        decoded = self.decode_seq(source)
        self.data.decoder_tokenizer.view_data((target, decoded))

    def save(self, name="s2s"):
        self.combined.save(name + '.h5')

    def load(self, name="s2s"):
        self.combined.load_weights(name + '.h5')

    def get_encoder(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        embeddings = Embedding(self.data.num_encoder_tokens, self.embedding_dim)(encoder_inputs)
        encoder_outputs = self.encoder_layer(embeddings)
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
    def __init__(self, latent_dim, data, embedding_dim=100, optimizer='adam'):
        super().__init__(latent_dim, data, embedding_dim)
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = Input(shape=(None,))
        _, e_state_h, e_state_c = self.encoder(encoder_inputs)
        decoded_output, state_h, state_c = self.decoder([decoder_inputs, e_state_h, e_state_c])
        self.combined = Model([encoder_inputs, decoder_inputs], decoded_output)
        self.combined.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
        # self.auto_encoder.summary()

    def get_decoder(self):
        decoder_inputs = Input(shape=(None,))
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        embeddings = Embedding(self.data.num_decoder_tokens, self.embedding_dim)(decoder_inputs)
        decoder_outputs, state_h, state_c = self.decoder_layer(embeddings,
                                                         initial_state=[decoder_state_input_h, decoder_state_input_c])
        decoder_dense = Dense(self.data.num_decoder_tokens, activation='softmax')
        decoded_output = TimeDistributed(decoder_dense)(decoder_outputs)
        decoder_model = Model([decoder_inputs, decoder_state_input_h, decoder_state_input_c],
                              [decoded_output, state_h, state_c])
        decoder_model.summary()
        return decoder_model

    def decode_seq(self, input_seq):
        e_out, dsh, dsc = self.encoder.predict(input_seq)
        target_seq = np.ones((input_seq.shape[0], 1))
        target_seq = target_seq * [self.data.decoder_tokenizer.start_tkn]
        # Sampling loop for a batch of sequences
        decoded = None
        for _ in range(self.data.max_decoder_seq_length):
            output_tokens, dsh, dsc = self.decoder.predict([target_seq, dsh, dsc])
            sampled = np.argmax(output_tokens, axis=2)
            decoded = sampled if decoded is None else np.hstack((decoded, sampled))
            target_seq = sampled
        return decoded


class Seq2seqAttention(EncoderDecoder):

    def __init__(self, latent_dim, data, embedding_dim=100, optimizer='adam'):
        super().__init__(latent_dim, data, embedding_dim)
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = Input(shape=(None,))
        encoder_outputs = self.encoder(encoder_inputs)
        decoded_output, state_h, state_c = self.decoder([decoder_inputs] + encoder_outputs)
        self.combined = Model([encoder_inputs, decoder_inputs], decoded_output)
        self.combined.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    def get_decoder(self):
        decoder_inputs = Input(shape=(None,))
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        encoder_outputs = Input(shape=(self.data.max_encoder_seq_length, self.latent_dim))
        embeddings = Embedding(self.data.num_decoder_tokens, self.embedding_dim)(decoder_inputs)
        decoder_outputs, state_h, state_c = self.decoder_layer(embeddings,
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
        target_seq = np.ones((input_seq.shape[0], 1))
        target_seq = target_seq * [self.data.decoder_tokenizer.start_tkn]
        # Sampling loop for a batch of sequences
        decoded = None
        for _ in range(self.data.max_decoder_seq_length):
            output_tokens, dsh, dsc = self.decoder.predict([target_seq, e_out, dsh, dsc])
            sampled = np.argmax(output_tokens, axis=2)
            decoded = sampled if decoded is None else np.hstack((decoded, sampled))
            target_seq = sampled
        return decoded


class BiSeq2seqAttention(Seq2seqAttention):

    def __init__(self, latent_dim, data, embedding_dim=100, optimizer='adam'):
        super().__init__(latent_dim, data, embedding_dim,optimizer=optimizer)

    def get_encoder(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        embeddings = Embedding(self.data.num_encoder_tokens, self.embedding_dim)(encoder_inputs)
        encoder_outputs, esh, esc, esh1, esc1 = Bidirectional(self.encoder_layer, merge_mode="mul")(embeddings)
        encoder_model = Model(encoder_inputs, [encoder_outputs, K.dot(esh, esh1), K.dot(esc, esc1)])
        encoder_model.summary()
        return encoder_model

