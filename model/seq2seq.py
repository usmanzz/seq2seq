from __future__ import print_function
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
from .components import Encoder, Decoder, BidirectionalEncoder


class Seq2seqAttention:
    def __init__(self, latent_dim, data, embedding_dim=100, optimizer="rmsprop"):
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.data = data
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        encoder_inputs = Input(shape=(self.data.max_encoder_seq_length,))
        decoder_inputs = Input(shape=(None,))
        outputs = self.encoder(encoder_inputs)
        decoded_output, states = self.decoder(*tuple(decoder_inputs) + outputs)
        self.combined = Model([encoder_inputs, decoder_inputs], decoded_output)
        self.combined.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

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
        return Encoder(self.data.num_encoder_tokens, self.embedding_dim, self.latent_dim)

    def get_decoder(self):
        return Decoder(self.data.num_decoder_tokens, self.embedding_dim, self.latent_dim)

    def decode_seq(self, input_seq):
        e_out, states = self.encoder.predict(input_seq)
        target_seq = np.ones((input_seq.shape[0], 1))
        target_seq = target_seq * [self.data.decoder_tokenizer.start_tkn]
        # Sampling loop for a batch of sequences
        decoded = None
        for _ in range(self.data.max_decoder_seq_length + 1):
            output_tokens, states = self.decoder.predict([target_seq, e_out, states])
            sampled = np.argmax(output_tokens, axis=2)
            decoded = sampled if decoded is None else np.hstack((decoded, sampled))
            target_seq = sampled
        return decoded


class BiSeq2seqAttention(Seq2seqAttention):

    def __init__(self, latent_dim, data, embedding_dim=100, optimizer='adam'):
        super().__init__(latent_dim, data, embedding_dim, optimizer)

    def get_encoder(self):
        return BidirectionalEncoder(self.data.num_encoder_tokens, self.embedding_dim, self.latent_dim)

    def get_decoder(self):
        return Decoder(self.data.num_encoder_tokens, self.embedding_dim, self.latent_dim*2)
