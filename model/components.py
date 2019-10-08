from __future__ import print_function
import tensorflow as tf
tf.executing_eagerly()

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        # self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x):
        x = self.embedding(x)
        outputs, state = self.gru(x)
        return outputs, state


class BidirectionalEncoder(Encoder):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(BidirectionalEncoder, self).__init__(vocab_size, embedding_dim, enc_units)
        self.bidirectional_gru = tf.keras.layers.Bidirectional(self.gru)

    def call(self, x):
        x = self.embedding(x)
        outputs, state1, state2 = self.bidirectional_gru(x)
        return outputs, tf.keras.layers.concatenate([state1, state2])


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        # self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.attention = tf.keras.layers.Attention()
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))

    def call(self, inp):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x, enc_output, init_state = inp
        x = self.embedding(x)
        decoder_outputs, state = self.gru(x, initial_state=[init_state])
        context_vector = self.attention([decoder_outputs, enc_output])
        outputs = tf.keras.layers.concatenate([context_vector, decoder_outputs])
        out = self.fc(outputs)
        return out, state

if __name__ == '__main__':
    encoder = BidirectionalEncoder(1000, 50, 200)
    decoder = Decoder(2355, 50, 400)
    encoder_inputs = tf.keras.layers.Input(shape=(13,))
    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    output, encoder_states = encoder(encoder_inputs)
    print(output.shape, encoder_states.shape)
    decoded_output, states = decoder([decoder_inputs, output, encoder_states])
    print(decoded_output.shape, states.shape)
    combined = tf.keras.Model([encoder_inputs, decoder_inputs], decoded_output)
    combined.compile(optimizer="adam", loss='sparse_categorical_crossentropy')
    combined.summary()
    # encoder.summary()
    decoder.summary()
