from __future__ import print_function
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Concatenate, LSTM, Embedding, GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Model
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

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

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
        output, state = self.gru(x)
        return output, state

    # def initialize_hidden_state(self):
    #     return tf.zeros((self.batch_sz, self.enc_units))


class BidirectionalEncoder(Encoder):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(BidirectionalEncoder, self).__init__(vocab_size, embedding_dim, enc_units)
        self.bidirectional_gru = Bidirectional(self.gru)

    def call(self, x):
        x = self.embedding(x)
        output, state1, state2 = self.bidirectional_gru(x)
        return output, Concatenate([state1, state2])

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
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, enc_output, states):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x, initial_state=states)

        context_vector, attention_weights = self.attention(output, enc_output)

        # output shape == (batch_size * 1, hidden_size)
        # output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        out = self.fc(context_vector)

        return out, state
