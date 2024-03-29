from __future__ import print_function
import tensorflow as tf
import numpy as np
from .components import Encoder, Decoder, BidirectionalEncoder, sparse_categorical_crossentropy


class Seq2seqAttention:
    def __init__(self, latent_dim, data, embedding_dim=100, optimizer="adam"):
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.data = data
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        encoder_inputs = tf.keras.layers.Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = tf.keras.layers.Input(shape=(None,))
        output, hidden = self.encoder(encoder_inputs)
        decoded_output, states = self.decoder([decoder_inputs, output, hidden])
        self.combined = tf.keras.Model([encoder_inputs, decoder_inputs], decoded_output)
        self.combined.compile(optimizer=optimizer, loss=sparse_categorical_crossentropy)
        self.inference = self.get_inference()
        self.combined.summary()
        # self.decoder.summary()

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
        self.combined.save_weights(name + '.h5')

    def load(self, name="s2s"):
        self.combined.load_weights(name + '.h5')

    def get_encoder(self):
        return Encoder(self.data.num_encoder_tokens, self.embedding_dim, self.latent_dim)

    def get_decoder(self):
        return Decoder(self.data.num_decoder_tokens, self.embedding_dim, self.latent_dim)

    def get_inference(self):
        encoder_inputs = tf.keras.layers.Input(shape=(self.data.max_encoder_seq_length,))
        e_out, states = self.encoder(encoder_inputs)
        target_seq = tf.ones((tf.shape(encoder_inputs)[0], 1), dtype=tf.dtypes.int64) * self.data.decoder_tokenizer.start_tkn
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
        super().__init__(latent_dim, data, embedding_dim, optimizer)

    def get_encoder(self):
        return BidirectionalEncoder(self.data.num_encoder_tokens, self.embedding_dim, self.latent_dim)

    def get_decoder(self):
        return Decoder(self.data.num_decoder_tokens, self.embedding_dim, self.latent_dim * 2)
