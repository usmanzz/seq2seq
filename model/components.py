from __future__ import print_function
from tensorflow.keras.layers import Input, Dense, TimeDistributed, concatenate, Embedding, GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Model
# from ..layers.attention import AttentionLayer
from tensorflow_core.python.keras.layers.wrappers import Bidirectional

from layers.attention import AttentionLayer
# import numpy as np
import tensorflow as tf


# from abc import ABCMeta, abstractmethod
# import tensorflow.keras.backend as K

# from ..layers.attention import AttentionLayer

class BahdanauAttention(Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, inputs):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        query, values = inputs

        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        # self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.enc_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')

    def call(self, x):
        x = self.embedding(x)
        outputs, state = self.gru(x)
        return outputs, state

    # def initialize_hidden_state(self):
    #     return tf.zeros((self.batch_sz, self.enc_units))


class BidirectionalEncoder(Encoder):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(BidirectionalEncoder, self).__init__(vocab_size, embedding_dim, enc_units)
        self.bidirectional_gru = Bidirectional(self.gru)

    def call(self, x):
        x = self.embedding(x)
        outputs, state1, state2 = self.bidirectional_gru(x)
        return outputs, concatenate([state1, state2])

    # def initialize_hidden_state(self):
    #     return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        # self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.fc = Dense(vocab_size)

        # used for attention
        # self.attention = BahdanauAttention(self.dec_units)
        self.attention = AttentionLayer()

    def call(self, x, enc_output, init_state):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # context_vector, attention_weights = self.attention(x, enc_output)
        # context_vectors = Concatenate()([context_vector, x])
        # passing the concatenated vector to the GRU
        decoder_outputs, state = self.gru(x, initial_state=[init_state])

        # context_vector, attention_weights = self.attention([enc_output, decoder_outputs])
        context_vector, attention_weights = self.attention([enc_output, decoder_outputs])

        # output shape == (batch_size * 1, hidden_size)
        # output = tf.reshape(output, (-1, output.shape[2]))
        outputs = concatenate([context_vector, decoder_outputs])
        # print(context_vector.shape, decoder_outputs.shape, outputs.shape)

        # output shape == (batch_size, vocab)
        out = TimeDistributed(self.fc)(outputs)

        return out, state


if __name__ == '__main__':
    encoder = BidirectionalEncoder(1000, 50, 200)
    decoder = Decoder(2355, 50, 400)
    encoder_inputs = Input(shape=(13,))
    decoder_inputs = Input(shape=(None,))
    output, encoder_states = encoder(encoder_inputs)
    print(output.shape, encoder_states.shape)
    decoded_output, states = decoder(decoder_inputs, output, encoder_states)
    print(decoded_output.shape, states.shape)
    combined = Model([encoder_inputs, decoder_inputs], decoded_output)
    combined.compile(optimizer="adam", loss='sparse_categorical_crossentropy')
    combined.summary()
    # encoder.summary()
    # decoder.summary()
