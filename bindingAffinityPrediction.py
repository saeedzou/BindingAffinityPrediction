import keras
from keras.layers import Dense, Concatenate, Embedding, TextVectorization, Dropout, Bidirectional, LSTM, Input, Add
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging, argparse, pickle, os
import tensorflow as tf

# Set the random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

def load_csv(path):
    return pd.read_csv(path, index_col=0)

def log_path(args, type, format):
    return (f'./log/{type}_e{args.epochs}_bs{args.batch_size}_lr{args.learning_rate}'
            f'_loss{args.loss}_ed{args.embedding_dim}_ru{args.rnn_units}_sl{args.seq_len}'
            f'_cd{args.context_dim}_vs{args.vocab_size}_fci{args.fc_in_units}_fco{args.fc_out_units}.{format}')

def adapt_vectorizer(vec, data):
    vec.adapt(data)
    return vec

def display_history(hist):
    plt.plot(hist['loss'], label='Training Loss')
    plt.plot(hist['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig('./data/loss.png')
    plt.show()
    plt.plot(hist['accuracy'], label='Training Accuracy')
    plt.plot(hist['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig('./data/accuracy.png')
    plt.show()
    plt.plot(hist['recall'], label='Training Recall')
    plt.plot(hist['val_recall'], label='Validation Recall')
    plt.legend()
    plt.title('Recall')
    plt.savefig('./data/recall.png')
    plt.show()
    plt.plot(hist['precision'], label='Training Precision')
    plt.plot(hist['val_precision'], label='Validation Precision')
    plt.legend()
    plt.title('Precision')
    plt.savefig('./data/precision.png')
    plt.show()
    plt.plot(hist['auc'], label='Training AUC')
    plt.plot(hist['val_auc'], label='Validation AUC')
    plt.legend()
    plt.title('AUC')
    plt.savefig('./data/auc.png')
    plt.show()

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, mhc_vec, pep_vec, batch_size=64, shuffle=True):
        self.data = data
        self.mhc_vec = mhc_vec
        self.pep_vec = pep_vec
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch = self.data.iloc[indices]
        mhc = self.mhc_vec(batch['MHC_sequence'].values)
        pep = self.pep_vec(batch['peptide_sequence'].values)
        label = batch['label'].values
        return [mhc, pep], label
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

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
    
def bindingPrediction(
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='./data/train.csv', help='Path to training data')
    parser.add_argument('--val', type=str, default='./data/val.csv', help='Path to validation data')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-l', '--loss', type=str, default='binary_crossentropy', help='Loss function')
    parser.add_argument('-ed', '--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('-ru', '--rnn_units', type=int, default=32, help='Number of RNN units')
    parser.add_argument('-s', '--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('-v', '--vocab_size', type=int, default=22, help='Vocabulary size')
    parser.add_argument('-fci', '--fc_in_units', type=int, default=32, help='Number of units in inner fully connected layer')
    parser.add_argument('-fco', '--fc_out_units', type=int, default=32, help='Number of units in outer fully connected layer')
    parser.add_argument('-cd', '--context_dim', type=int, default=16, help='Context dimension')
    args = parser.parse_args()
    log_path = log_path(args, 'train', 'log')
    logging.basicConfig(
        level=logging.INFO, filename=log_path, filemode='w', 
        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Logging Started')
    logging.info('Loading Training Data')
    train = load_csv(args.train)
    logging.info('Loading Validation Data')
    val = load_csv(args.val)


    if not os.path.exists('./data/mhc_vec.pkl'):
        mhc_vec = TextVectorization(
            split='character', output_mode='int', 
            output_sequence_length=args.seq_len, standardize=None)
        logging.info('Fitting MHC Vectorizer')
        mhc_vec = adapt_vectorizer(mhc_vec, train['MHC_sequence'].values)
        with open('./data/mhc_vec.pkl', 'wb') as f:
            pickle.dump(
                {'config': mhc_vec.get_config(), 
                'weights': mhc_vec.get_weights()}, 
                f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        logging.info('Loading MHC Vectorizer')
        with open('./data/mhc_vec.pkl', 'rb') as f:
            info = pickle.load(f)
        mhc_vec = TextVectorization.from_config(info['config'])
        mhc_vec.set_weights(info['weights'])

    if not os.path.exists('./data/pep_vec.pkl'):
        pep_vec = TextVectorization(
            split='character', output_mode='int', 
            output_sequence_length=args.seq_len, standardize=None)
        logging.info('Fitting Peptide Vectorizer')
        pep_vec = adapt_vectorizer(pep_vec, train['peptide_sequence'].values)
        with open('./data/pep_vec.pkl', 'wb') as f:
            pickle.dump(
                {'config': pep_vec.get_config(), 
                 'weights': pep_vec.get_weights()}, 
                f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        logging.info('Loading Peptide Vectorizer')
        with open('./data/pep_vec.pkl', 'rb') as f:
            info = pickle.load(f)
        pep_vec = TextVectorization.from_config(info['config'])
        pep_vec.set_weights(info['weights'])
    
    logging.info('Creating Model')
    model = bindingPrediction(embedding_dim=args.embedding_dim,
                                rnn_units=args.rnn_units,
                                seq_len=args.seq_len,
                                context_dim=args.context_dim,
                                vocab_size=args.vocab_size,
                                fc_in_units=args.fc_in_units,
                                fc_out_units=args.fc_out_units)
    logging.info('Model Summary')
    model.build(input_shape=[(None, 1), (None, 1)])
    keras.utils.plot_model(model, to_file='./data/model.png', show_shapes=True, dpi=512)
    summary_buffer = []
    model.summary(print_fn=lambda x: summary_buffer.append(x))
    logging.info('\n'.join(summary_buffer))
    metrics = [keras.metrics.BinaryAccuracy(name='accuracy'), 
            keras.metrics.Recall(name='recall'), 
            keras.metrics.Precision(name='precision'), 
            keras.metrics.AUC(name='auc')]
    lr = args.learning_rate
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    logging.info(f'Compiling Model | Metrics: {[m.name for m in metrics]} '
                 f'| Learning Rate: {lr} | Loss: {args.loss}')
    model.compile(optimizer='adam', loss=args.loss, metrics=metrics)

    logging.info('Training Model')
    train_data_generator = DataGenerator(train, mhc_vec, pep_vec, batch_size=args.batch_size)
    val_data_generator = DataGenerator(val, mhc_vec, pep_vec, batch_size=args.batch_size, shuffle=False)
    h = model.fit(train_data_generator, 
                  epochs=args.epochs, 
                  batch_size=args.batch_size, 
                  validation_data=val_data_generator)
    
    logging.info('Saving Model')
    model_path = log_path(args, 'model', 'h5')
    model.save(model_path)
    logging.info('Saving History')
    hist = pd.DataFrame(h.history)
    hist_path = log_path(args, 'history', 'csv')
    hist.to_csv(hist_path)
    logging.info('Done')

    logging.info('Plotting History')
    display_history(hist)

    logging.info('Evaluating Model')
    # evaluate on training data
    logging.info('On Training Data:')
    model.evaluate(train_data_generator)
    # evaluate on validation data
    logging.info('On Validation Data:')
    model.evaluate(val_data_generator)
    
    logging.info('Done')
    logging.info('Logging Ended')