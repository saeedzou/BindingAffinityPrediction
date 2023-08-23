import joblib, os, argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import scipy.sparse as sps

def train(train_csv, mhc_vec_path, pep_vec_path, model):
    # Read csv file
    print('-- Reading csv file --')
    df = pd.read_csv(train_csv, index_col=0)
    if os.path.exists(mhc_vec_path):
        print('-- Loading mhc vectorizer --')
        mhc_vec = joblib.load(mhc_vec_path)
        X_mhc = mhc_vec.transform(df['mhc_sequence'])
    else:
        print('-- Training mhc vectorizer --')
        mhc_vec = TfidfVectorizer(lowercase=False, analyzer='char')
        X_mhc = mhc_vec.fit_transform(df['mhc_sequence'])
        print('-- Saving mhc vectorizer --')
        joblib.dump(mhc_vec, mhc_vec_path)
    if os.path.exists(pep_vec_path):
        print('-- Loading peptide vectorizer --')
        pep_vec = joblib.load(pep_vec_path)
        X_pep = pep_vec.transform(df['peptide_sequence'])
    else:
        print('-- Training peptide vectorizer --')
        pep_vec = TfidfVectorizer(lowercase=False, analyzer='char')
        X_pep = pep_vec.fit_transform(df['peptide_sequence'])
        print('-- Saving peptide vectorizer --')
        joblib.dump(pep_vec, pep_vec_path)
    X = sps.hstack([X_mhc, X_pep])
    y = np.array(df['label'])
    # Train model
    print('-- Training model --')
    model.fit(X, y)
    return model

def predict(test_csv, mhc_vec_path, pep_vec_path, model):
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
    X_mhc = mhc_vec.transform(df['mhc_sequence'])
    print('-- Loading peptide vectorizer --')
    pep_vec = joblib.load(pep_vec_path)
    X_pep = pep_vec.transform(df['peptide_sequence'])
    X = sps.hstack([X_mhc, X_pep])
    y = np.array(df['label'])
    # Predict
    print('-- Predicting --')
    y_pred = model.predict(X)
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
    args = parser.parse_args()
    # load model if exists
    if os.path.exists(args.model_path):
        print('-- Loading model --')
        model = joblib.load(args.model_path)
    else:
        print('-- Training model --')
        model = LinearSVC()
        model = train(args.train_csv, args.mhc_vec_path, args.pep_vec_path, model)
        print('-- Saving model --')
        joblib.dump(model, args.model_path)
    # predict
    predict(args.test_csv, args.mhc_vec_path, args.pep_vec_path, model)
