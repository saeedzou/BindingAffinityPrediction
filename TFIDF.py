import joblib, os, argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import scipy.sparse as sps
import tensorflow as tf
import keras
# import dense
from keras.layers import Dense, Dropout, Input, Concatenate

class customLayer(keras.layers.Layer):
    def __init__(self, hidden_dims=[32, 64], activation=None, **kwargs):
        super(customLayer, self).__init__()
        self.hidden_dims = hidden_dims
        self.activation = keras.activations.get(activation)
        self.denses = [Dense(dim, activation=self.activation) for dim in self.hidden_dims]
    
    def call(self, inputs):
        x = inputs
        for dense in self.denses:
            x = dense(x)
        return x

class net(keras.Model):
    def __init__(self, **kwargs):
        super(net, self).__init__()
        self.mhc = customLayer(activation='relu')
        self.pep = customLayer(activation='relu')
        self.concat = Concatenate(axis=1)
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        mhc, pep = inputs
        mhc = self.mhc(mhc)
        pep = self.pep(pep)
        x = self.concat([mhc, pep])
        x = self.dense(x)
        return x


def train(train_csv, mhc_vec_path, pep_vec_path, model_path, lr=1e-3, epochs=5):
    # Read csv file
    print('-- Reading csv file --')
    df = pd.read_csv(train_csv, index_col=0)
    if os.path.exists(mhc_vec_path):
        print('-- Loading mhc vectorizer --')
        mhc_vec = joblib.load(mhc_vec_path)
        X_mhc = mhc_vec.transform(df['MHC_sequence']).todense()
    else:
        print('-- Training mhc vectorizer --')
        mhc_vec = TfidfVectorizer(lowercase=False, analyzer='char')
        X_mhc = mhc_vec.fit_transform(df['MHC_sequence']).todense()
        print('-- Saving mhc vectorizer --')
        joblib.dump(mhc_vec, mhc_vec_path)
    if os.path.exists(pep_vec_path):
        print('-- Loading peptide vectorizer --')
        pep_vec = joblib.load(pep_vec_path)
        X_pep = pep_vec.transform(df['peptide_sequence']).todense()
    else:
        print('-- Training peptide vectorizer --')
        pep_vec = TfidfVectorizer(lowercase=False, analyzer='char')
        X_pep = pep_vec.fit_transform(df['peptide_sequence']).todense()
        print('-- Saving peptide vectorizer --')
        joblib.dump(pep_vec, pep_vec_path)
    y = np.array(df['label'])
    # Train model if not exists
    if os.path.exists(model_path):
        print('-- Loading model --')
        model = keras.models.load_model(model_path)
    else:
        print('-- Training model --')
        model = net()
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit([X_mhc, X_pep], y, epochs=epochs, batch_size=64)
        print('-- Saving model --')
        model.save(model_path)
    return model

def predict(test_csv, mhc_vec_path, pep_vec_path, model_path):
    # Read csv file
    print('-- Reading csv file --')
    df = pd.read_csv(test_csv, index_col=0)
    # raise exception if vectorizer not found
    if not os.path.exists(mhc_vec_path):
        raise Exception('MHC vectorizer not found')
    if not os.path.exists(pep_vec_path):
        raise Exception('Peptide vectorizer not found')
    # Load vectorizer
    print('-- Loading mhc vectorizer --')
    mhc_vec = joblib.load(mhc_vec_path)
    X_mhc = mhc_vec.transform(df['MHC_sequence']).todense()
    print('-- Loading peptide vectorizer --')
    pep_vec = joblib.load(pep_vec_path)
    X_pep = pep_vec.transform(df['peptide_sequence']).todense()
    y = np.array(df['label'])
    # load model
    print('-- Loading model --')
    model = keras.models.load_model(model_path)
    # Predict
    print('-- Predicting --')
    y_pred = model.predict([X_mhc, X_pep])
    y_pred = np.round(y_pred)
    # Report
    print('-- Report --')
    print(classification_report(y, y_pred))
    return y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and predict using TFIDF')
    parser.add_argument('-t', '--train_csv', default='data/train.csv', help='Path to train csv file')
    parser.add_argument('-v', '--test_csv', default='data/val.csv', help='Path to test csv file')
    parser.add_argument('-m', '--mhc_vec_path', default='data/mhc_vec.pkl', help='Path to mhc vectorizer')
    parser.add_argument('-p', '--pep_vec_path', default='data/pep_vec.pkl', help='Path to peptide vectorizer')
    parser.add_argument('-o', '--model_path', default='data/tfidf_model.pkl', help='Path to model')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs')
    args = parser.parse_args()
    # train
    model = train(args.train_csv, args.mhc_vec_path, args.pep_vec_path, args.model_path, args.learning_rate, args.epochs)
    # predict
    predict(args.test_csv, args.mhc_vec_path, args.pep_vec_path, args.model_path)
