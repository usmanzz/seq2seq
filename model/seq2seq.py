from __future__ import print_function
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Concatenate, Embedding, GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Model
import numpy as np
from abc import ABCMeta, abstractmethod
from ..layers.attention import AttentionLayer


class EncoderDecoder(metaclass=ABCMeta):
    def __init__(self, enc_units, dec_units, data, embedding_dim=100):
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.embedding_dim = embedding_dim
        self.data = data
        self.encoder_layer = GRU(self.enc_units, return_state=True, return_sequences=True)
        self.decoder_layer = GRU(self.dec_units, return_sequences=True, return_state=True)
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
    def __init__(self, enc_units, dec_units, data, embedding_dim=100, optimizer='adam'):
        super().__init__(enc_units, dec_units, data, embedding_dim)
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = Input(shape=(None,))
        _, e_state = self.encoder(encoder_inputs)
        decoded_output, state = self.decoder([decoder_inputs, e_state])
        self.combined = Model([encoder_inputs, decoder_inputs], decoded_output)
        self.combined.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
        # self.auto_encoder.summary()

    def get_decoder(self):
        decoder_inputs = Input(shape=(None,))
        encoder_state = Input(shape=(self.enc_units,))
        embeddings = Embedding(self.data.num_decoder_tokens, self.embedding_dim)(decoder_inputs)
        decoder_outputs, decoder_state = self.decoder_layer(embeddings,
                                                            initial_state=encoder_state)
        decoder_dense = Dense(self.data.num_decoder_tokens, activation='softmax')
        decoded_output = TimeDistributed(decoder_dense)(decoder_outputs)
        decoder_model = Model([decoder_inputs, encoder_state],
                              [decoded_output, decoder_state])
        decoder_model.summary()
        return decoder_model

    def decode_seq(self, input_seq):
        e_out, state = self.encoder.predict(input_seq)
        target_seq = np.ones((input_seq.shape[0], 1))
        target_seq = target_seq * [self.data.decoder_tokenizer.start_tkn]
        # Sampling loop for a batch of sequences
        decoded = None
        for _ in range(self.data.max_decoder_seq_length):
            output_tokens, state = self.decoder.predict([target_seq, state])
            sampled = np.argmax(output_tokens, axis=2)
            decoded = sampled if decoded is None else np.hstack((decoded, sampled))
            target_seq = sampled
        return decoded


class Seq2seqAttention(EncoderDecoder):

    def __init__(self, enc_units, dec_units, data, embedding_dim=100, optimizer='adam'):
        super().__init__(enc_units, dec_units, data, embedding_dim)
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = Input(shape=(None,))
        encoder_outputs = self.encoder(encoder_inputs)
        decoded_output, state = self.decoder([decoder_inputs] + encoder_outputs)
        self.combined = Model([encoder_inputs, decoder_inputs], decoded_output)
        self.combined.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    def get_decoder(self):
        decoder_inputs = Input(shape=(None,))
        encoder_state = Input(shape=(self.dec_units,))
        encoder_outputs = Input(shape=(self.data.max_encoder_seq_length, self.dec_units))
        embeddings = Embedding(self.data.num_decoder_tokens, self.embedding_dim)(decoder_inputs)
        decoder_outputs, state = self.decoder_layer(embeddings, initial_state=encoder_state)
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        context_vectors = Concatenate()([attn_out, decoder_outputs])
        decoder_dense = Dense(self.data.num_decoder_tokens, activation='softmax')
        decoded_output = TimeDistributed(decoder_dense)(context_vectors)
        decoder_model = Model([decoder_inputs, encoder_outputs, encoder_state],
                              [decoded_output, state])
        decoder_model.summary()
        return decoder_model

    def decode_seq(self, input_seq):
        e_out, state = self.encoder.predict(input_seq)
        target_seq = np.ones((input_seq.shape[0], 1))
        target_seq = target_seq * [self.data.decoder_tokenizer.start_tkn]
        # Sampling loop for a batch of sequences
        decoded = None
        for _ in range(self.data.max_decoder_seq_length + 1):
            output_tokens, dsh, dsc = self.decoder.predict([target_seq, e_out, state])
            sampled = np.argmax(output_tokens, axis=2)
            decoded = sampled if decoded is None else np.hstack((decoded, sampled))
            target_seq = sampled
        return decoded


class BiSeq2seqAttention(Seq2seqAttention):

    def __init__(self, enc_units, dec_units, data, embedding_dim=100, optimizer='adam'):
        super().__init__(enc_units, dec_units*2, data, embedding_dim)

    def get_encoder(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        embeddings = Embedding(self.data.num_encoder_tokens, self.embedding_dim)(encoder_inputs)
        encoder_outputs, forward, backward = Bidirectional(self.encoder_layer)(embeddings)
        encoder_model = Model(encoder_inputs, [encoder_outputs, Concatenate()([forward, backward])])
        encoder_model.summary()
        return encoder_model

