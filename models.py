import numpy as np
import keras
from keras.layers import Dense, Concatenate, Embedding, Dropout, Bidirectional, LSTM, Input, Add, GlobalAveragePooling1D
import tensorflow as tf
# Set the random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

class Attention(keras.layers.Layer):
    def __init__(self, feature_dim, step_dim, context_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.context_dim = context_dim

    def build(self, input_shape):
        self.tanh = keras.layers.Activation('tanh')
        self.weight = self.add_weight(
            shape=(self.feature_dim, self.context_dim), 
            initializer=keras.initializers.GlorotUniform(), 
            trainable=True, 
            name="attention_weight")
        self.b = self.add_weight(
            shape=(self.step_dim, self.context_dim), 
            initializer='zeros', 
            trainable=True, 
            name="attention_b")
        self.context_vector = self.add_weight(
            shape=(self.context_dim, 1), 
            initializer=keras.initializers.GlorotUniform(), 
            trainable=True, 
            name="attention_context_vector")
        super(Attention, self).build(input_shape)

    def call(self, x):
        eij = keras.backend.dot(x, self.weight)
        eij = self.tanh(keras.backend.bias_add(eij, self.b))
        v = keras.backend.exp(keras.backend.dot(eij, self.context_vector))
        v = v / (keras.backend.sum(v, axis=1, keepdims=True))
        weighted_input = x * v
        s = keras.backend.sum(weighted_input, axis=1)
        return s
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            'feature_dim': self.feature_dim, 
            'step_dim': self.step_dim, 
            'context_dim': self.context_dim})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def mhcAttentionSimple(
        embedding_dim=64, 
        rnn_units=32, 
        seq_len=100, 
        context_dim=16, 
        vocab_size=22, 
        fc_in_units=32, 
        fc_out_units=32):
    
    mhc_input = Input(shape=(seq_len,), dtype='int32', name='MHC-sequence')
    pep_input = Input(shape=(seq_len,), dtype='int32', name='peptide-sequence')

    mhc_emb = Embedding(vocab_size, embedding_dim, name='MHC-embedding')(mhc_input)
    mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='MHC-LSTM-1'), name='MHC-BiLSTM-1')(mhc_emb)
    mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='MHC-LSTM-2'), name='MHC-BiLSTM-2')(mhc_rnn)
    mhc_att = Attention(embedding_dim, seq_len, context_dim, name='MHC-attention')(mhc_rnn)
    mhc = Dense(fc_in_units, activation='relu', name='MHC-FC')(mhc_att)

    pep_emb = Embedding(vocab_size, embedding_dim, name='peptide-embedding')(pep_input)
    pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='peptide-LSTM-1'), name='peptide-BiLSTM-1')(pep_emb)
    pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='peptide-LSTM-2'), name='peptide-BiLSTM-2')(pep_rnn)
    pep_att = Attention(embedding_dim, seq_len, context_dim, name='peptide-attention')(pep_rnn)
    pep = Dense(fc_in_units, activation='relu', name='peptide-FC')(pep_att)

    x = Concatenate(axis=1)([mhc, pep])
    x = Dense(fc_out_units, activation='relu', name='concat-FC')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid', name='output')(x)
    model = keras.Model(inputs=[mhc_input, pep_input], outputs=x)
    return model

def mhcNoAttention(
        embedding_dim=64, 
        rnn_units=32, 
        seq_len=100, 
        context_dim=16, 
        vocab_size=22, 
        fc_in_units=32, 
        fc_out_units=32):
    
    mhc_input = Input(shape=(seq_len,), dtype='int32', name='MHC-sequence')
    pep_input = Input(shape=(seq_len,), dtype='int32', name='peptide-sequence')

    mhc_emb = Embedding(vocab_size, embedding_dim, name='MHC-embedding')(mhc_input)
    mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='MHC-LSTM-1'), name='MHC-BiLSTM-1')(mhc_emb)
    mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='MHC-LSTM-2'), name='MHC-BiLSTM-2')(mhc_rnn)
    mhc = GlobalAveragePooling1D()(mhc_rnn)
    mhc = Dense(fc_in_units, activation='relu', name='MHC-FC')(mhc)

    pep_emb = Embedding(vocab_size, embedding_dim, name='peptide-embedding')(pep_input)
    pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='peptide-LSTM-1'), name='peptide-BiLSTM-1')(pep_emb)
    pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='peptide-LSTM-2'), name='peptide-BiLSTM-2')(pep_rnn)
    pep = GlobalAveragePooling1D()(pep_rnn)
    pep = Dense(fc_in_units, activation='relu', name='peptide-FC')(pep)

    x = Concatenate(axis=1)([mhc, pep])
    x = Dense(fc_out_units, activation='relu', name='concat-FC')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid', name='output')(x)
    model = keras.Model(inputs=[mhc_input, pep_input], outputs=x)
    return model

def mhcAttentionAdd(
        embedding_dim=64, 
        rnn_units=32, 
        seq_len=100, 
        context_dim=16, 
        vocab_size=22, 
        fc_in_units=32, 
        fc_out_units=32):
    
    mhc_input = Input(shape=(seq_len,), dtype='int32', name='MHC-sequence')
    pep_input = Input(shape=(seq_len,), dtype='int32', name='peptide-sequence')

    mhc_emb = Embedding(vocab_size, embedding_dim, name='MHC-embedding')(mhc_input)
    mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='MHC-LSTM-1'), name='MHC-BiLSTM-1')(mhc_emb)
    mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='MHC-LSTM-2'), name='MHC-BiLSTM-2')(mhc_rnn)
    mhc_att = Attention(embedding_dim, seq_len, context_dim, name='MHC-attention')(mhc_rnn)
    mhc = Dense(fc_in_units, activation='relu', name='MHC-FC')(mhc_att)

    pep_emb = Embedding(vocab_size, embedding_dim, name='peptide-embedding')(pep_input)
    pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='peptide-LSTM-1'), name='peptide-BiLSTM-1')(pep_emb)
    pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='peptide-LSTM-2'), name='peptide-BiLSTM-2')(pep_rnn)
    pep_att = Attention(embedding_dim, seq_len, context_dim, name='peptide-attention')(pep_rnn)
    pep = Dense(fc_in_units, activation='relu', name='peptide-FC')(pep_att)

    mhc_pep = Add(name='MHC-peptide')([mhc_rnn, pep_rnn])
    mhc_pep = Attention(embedding_dim, seq_len, context_dim, name='MHC-peptide-attention')(mhc_pep)
    mhc_pep = Dense(fc_in_units, activation='relu', name='MHC-peptide-FC')(mhc_pep)

    x = Concatenate(axis=1)([mhc, mhc_pep, pep])
    x = Dense(fc_out_units, activation='relu', name='concat-FC')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid', name='output')(x)
    model = keras.Model(inputs=[mhc_input, pep_input], outputs=x)
    return model

def mhcAttentionConcat(
        embedding_dim=64, 
        rnn_units=32, 
        seq_len=100, 
        context_dim=16, 
        vocab_size=22, 
        fc_in_units=32, 
        fc_out_units=32):
    
    mhc_input = Input(shape=(seq_len,), dtype='int32', name='MHC-sequence')
    pep_input = Input(shape=(seq_len,), dtype='int32', name='peptide-sequence')

    mhc_emb = Embedding(vocab_size, embedding_dim, name='MHC-embedding')(mhc_input)
    mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='MHC-LSTM-1'), name='MHC-BiLSTM-1')(mhc_emb)
    mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='MHC-LSTM-2'), name='MHC-BiLSTM-2')(mhc_rnn)
    mhc_att = Attention(embedding_dim, seq_len, context_dim, name='MHC-attention')(mhc_rnn)
    mhc = Dense(fc_in_units, activation='relu', name='MHC-FC')(mhc_att)

    pep_emb = Embedding(vocab_size, embedding_dim, name='peptide-embedding')(pep_input)
    pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='peptide-LSTM-1'), name='peptide-BiLSTM-1')(pep_emb)
    pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='peptide-LSTM-2'), name='peptide-BiLSTM-2')(pep_rnn)
    pep_att = Attention(embedding_dim, seq_len, context_dim, name='peptide-attention')(pep_rnn)
    pep = Dense(fc_in_units, activation='relu', name='peptide-FC')(pep_att)

    mhc_pep = Concatenate(axis=1)([mhc_rnn, pep_rnn])
    mhc_pep = Attention(embedding_dim, 2*seq_len, context_dim, name='MHC-peptide-attention')(mhc_pep)
    mhc_pep = Dense(fc_in_units, activation='relu', name='MHC-peptide-FC')(mhc_pep)

    x = Concatenate(axis=1)([mhc, mhc_pep, pep])
    x = Dense(fc_out_units, activation='relu', name='concat-FC')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid', name='output')(x)
    model = keras.Model(inputs=[mhc_input, pep_input], outputs=x)
    return model

def mhcAttentionCombined(
        embedding_dim=64, 
        rnn_units=32, 
        seq_len=100, 
        context_dim=16, 
        vocab_size=22, 
        fc_in_units=32, 
        fc_out_units=32):
    
    mhc_input = Input(shape=(seq_len,), dtype='int32', name='MHC-sequence')
    pep_input = Input(shape=(seq_len,), dtype='int32', name='peptide-sequence')

    mhc_emb = Embedding(vocab_size, embedding_dim, name='MHC-embedding')(mhc_input)
    mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='MHC-LSTM-1'), name='MHC-BiLSTM-1')(mhc_emb)
    mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='MHC-LSTM-2'), name='MHC-BiLSTM-2')(mhc_rnn)

    pep_emb = Embedding(vocab_size, embedding_dim, name='peptide-embedding')(pep_input)
    pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='peptide-LSTM-1'), name='peptide-BiLSTM-1')(pep_emb)
    pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, name='peptide-LSTM-2'), name='peptide-BiLSTM-2')(pep_rnn)

    mhc_pep = Concatenate(axis=1)([mhc_rnn, pep_rnn])
    mhc_pep = Attention(embedding_dim, 2*seq_len, context_dim, name='MHC-peptide-attention')(mhc_pep)
    mhc_pep = Dense(fc_in_units, activation='relu', name='MHC-peptide-FC')(mhc_pep)

    x = Dense(fc_out_units, activation='relu', name='concat-FC')(mhc_pep)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid', name='output')(x)
    model = keras.Model(inputs=[mhc_input, pep_input], outputs=x)
    return model