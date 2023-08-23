import keras
from keras.layers import Dense, Concatenate, Embedding, TextVectorization, GlobalAveragePooling1D, Bidirectional, LSTM
import pandas as pd
import matplotlib.pyplot as plt
import logging, argparse, sys
from io import StringIO

def load_csv(path):
    return pd.read_csv(path, index_col=0)

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


class bindingPrediction(keras.Model):
    def __init__(self, embedding_dim=64, rnn_units=32, mhc_vec=None, pep_vec=None, **kwargs):
        super(bindingPrediction, self).__init__()
        self.mhc_vec = mhc_vec
        self.pep_vec = pep_vec
        self.mhc_emb = Embedding(len(mhc_vec.get_vocabulary()), embedding_dim)
        self.pep_emb = Embedding(len(pep_vec.get_vocabulary()), embedding_dim)
        self.concat = Concatenate(axis=1)
        self.fc1 = Dense(16, activation='relu')
        self.fc2 = Dense(1, activation='sigmoid')
        self.mhc_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True))
        self.pep_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True))


    def call(self, inputs):
        mhc, pep = inputs
        mhc = self.mhc_emb(self.mhc_vec(mhc))
        pep = self.pep_emb(self.pep_vec(pep))
        mhc = self.mhc_rnn(mhc)
        pep = self.pep_rnn(pep)
        x = self.concat([mhc, pep])
        x = GlobalAveragePooling1D()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


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
    args = parser.parse_args()
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    log_path = f'./data/log_epochs_{args.epochs}_batch_size_{args.batch_size}_lr_{args.learning_rate}_loss_{args.loss}_embedding_dim_{args.embedding_dim}_rnn_units_{args.rnn_units}_seq_len_{args.seq_len}.txt'
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Logging Started')
    logging.info('Loading Training Data')
    train = load_csv(args.train)
    logging.info('Loading Validation Data')
    val = load_csv(args.val)

    mhc_vec = TextVectorization(split='character', output_mode='int', output_sequence_length=args.seq_len)
    logging.info('Fitting MHC Vectorizer')
    mhc_vec = adapt_vectorizer(mhc_vec, train['MHC_sequence'].values)
    pep_vec = TextVectorization(split='character', output_mode='int', output_sequence_length=args.seq_len)
    logging.info('Fitting Peptide Vectorizer')
    pep_vec = adapt_vectorizer(pep_vec, train['peptide_sequence'].values)

    logging.info('Creating Model')
    model = bindingPrediction(embedding_dim=args.embedding_dim, rnn_units=args.rnn_units, mhc_vec=mhc_vec, pep_vec=pep_vec)
    logging.info('Model Summary')
    model.build(input_shape=[(None, 1), (None, 1)])
    model.summary()
    logging.info(sys.stdout.getvalue())
    sys.stdout = original_stdout
    metrics = [keras.metrics.BinaryAccuracy(name='accuracy'), 
            keras.metrics.Recall(name='recall'), 
            keras.metrics.Precision(name='precision'), 
            keras.metrics.AUC(name='auc')]
    lr = args.learning_rate
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    logging.info(f'Compiling Model | Metrics: {[m.name for m in metrics]} | Learning Rate: {lr} | Loss: {args.loss}')
    model.compile(optimizer='adam', loss=args.loss, metrics=metrics)

    logging.info('Training Model')
    h = model.fit([train['MHC_sequence'].values, 
                   train['peptide_sequence'].values], 
                  train['label'].values, 
                  epochs=args.epochs, 
                  batch_size=args.batch_size, 
                  validation_data=([val['MHC_sequence'].values, 
                                    val['peptide_sequence'].values], 
                                    val['label'].values))
    
    logging.info('Saving Model')
    model.save(f'./data/model_epochs_{args.epochs}_batch_size_{args.batch_size}_lr_{args.learning_rate}_loss_{args.loss}_embedding_dim_{args.embedding_dim}_rnn_units_{args.rnn_units}_seq_len_{args.seq_len}.h5')
    logging.info('Saving History')
    hist = pd.DataFrame(h.history)
    hist.to_csv(f'./data/train_history_epochs_{args.epochs}_batch_size_{args.batch_size}_lr_{args.learning_rate}_loss_{args.loss}_embedding_dim_{args.embedding_dim}_rnn_units_{args.rnn_units}_seq_len_{args.seq_len}.csv')
    logging.info('Done')

    logging.info('Plotting History')
    display_history(hist)

    logging.info('Evaluating Model')
    # evaluate on training data
    logging.info('On Training Data:')
    model.evaluate([train['MHC_sequence'].values,
                    train['peptide_sequence'].values],
                   train['label'].values)
    # evaluate on validation data
    logging.info('On Validation Data:')
    model.evaluate([val['MHC_sequence'].values,
                    val['peptide_sequence'].values],
                   val['label'].values)
    
    logging.info('Done')
    logging.info('Logging Ended')