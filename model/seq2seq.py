from __future__ import print_function
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Concatenate, Embedding, GRU, Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
# from ..layers.attention import AttentionLayer


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

class Seq2seqAttention(EncoderDecoder):

    def __init__(self, enc_units, dec_units, data, embedding_dim=100, optimizer='adam'):
        super().__init__(enc_units, dec_units, data, embedding_dim)
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = Input(shape=(None,))
        encoder_outputs = self.encoder(encoder_inputs)
        decoded_output, state = self.decoder([decoder_inputs] + encoder_outputs)
        self.combined = Model([encoder_inputs, decoder_inputs], decoded_output)
        self.combined.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
        self.inference = self.get_inference()

    def get_decoder(self):
        decoder_inputs = Input(shape=(None,))
        encoder_state = Input(shape=(self.dec_units,))
        encoder_outputs = Input(shape=(self.data.max_encoder_seq_length, self.dec_units))
        embeddings = Embedding(self.data.num_decoder_tokens, self.embedding_dim)(decoder_inputs)
        decoder_outputs, state = self.decoder_layer(embeddings, initial_state=encoder_state)
        # attn_layer = AttentionLayer(name='attention_layer')
        # attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        attention_layer = tf.keras.layers.AdditiveAttention()
        attention_output = attention_layer([decoder_outputs, encoder_outputs])
        context_vectors = Concatenate()([attention_output, decoder_outputs])
        decoder_dense = Dense(self.data.num_decoder_tokens, activation='softmax')
        decoded_output = TimeDistributed(decoder_dense)(context_vectors)
        decoder_model = Model([decoder_inputs, encoder_outputs, encoder_state],
                              [decoded_output, state])
        decoder_model.summary()
        return decoder_model

    def get_inference(self):
        encoder_inputs = tf.keras.layers.Input(shape=(self.data.max_encoder_seq_length,))
        e_out, states = self.encoder(encoder_inputs)
        target_seq = tf.ones((tf.shape(encoder_inputs)[0], 1),
                             dtype=tf.dtypes.int64) * self.data.decoder_tokenizer.start_tkn
        decoded = target_seq
        for _ in range(self.data.max_decoder_seq_length + 1):
            output_tokens, states = self.decoder([target_seq, e_out, states])
            target_seq = tf.argmax(output_tokens, axis=-1)
            decoded = tf.keras.layers.concatenate([decoded, target_seq])
        return tf.keras.Model(encoder_inputs, decoded)

    def decode_seq(self, input_seq):
        decoded = self.inference.predict(input_seq)
        return decoded


class BiSeq2seqAttention(Seq2seqAttention):

    def __init__(self, latent_dim, data, embedding_dim=100, optimizer='adam'):
        super().__init__(latent_dim, latent_dim*2, data, embedding_dim)

    def get_encoder(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        embeddings = Embedding(self.data.num_encoder_tokens, self.embedding_dim)(encoder_inputs)
        encoder_outputs, forward, backward = Bidirectional(self.encoder_layer)(embeddings)
        encoder_model = Model(encoder_inputs, [encoder_outputs, Concatenate()([forward, backward])])
        encoder_model.summary()
        return encoder_model

