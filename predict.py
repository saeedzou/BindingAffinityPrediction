import keras
from keras.layers import TextVectorization
import pandas as pd
import numpy as np
import logging, argparse, pickle, os
import tensorflow as tf
from utils import load_csv, path_formatter, adapt_vectorizer, display_history
from dataloader import DataGenerator
import h5py    
from models import Attention

# Set the random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='./data/test.csv', help='Path to data')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('-b', '--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-l', '--loss', type=str, default='binary_crossentropy', help='Loss function')
    parser.add_argument('-ed', '--embedding_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('-ru', '--rnn_units', type=int, default=128, help='Number of RNN units')
    parser.add_argument('-s', '--seq_len', type=int, default=100, help='Sequence length')
    parser.add_argument('-v', '--vocab_size', type=int, default=22, help='Vocabulary size')
    parser.add_argument('-fci', '--fc_in_units', type=int, default=64, help='Number of units in inner fully connected layer')
    parser.add_argument('-fco', '--fc_out_units', type=int, default=64, help='Number of units in outer fully connected layer')
    parser.add_argument('-cd', '--context_dim', type=int, default=16, help='Context dimension')
    parser.add_argument('-m', '--model', type=str, default='mhcAttentionAdd', 
                        help='Model to use. Options: mhcAttentionAdd, mhcAttentionCombined, mhcAttentionConcat, mhcAttentionSimple, mhcNoAttention')
    args = parser.parse_args()

    data = load_csv(args.data)
    if os.path.exists('./data/mhc_vec.pkl'):
        with open('./data/mhc_vec.pkl', 'rb') as f:
            info = pickle.load(f)
        mhc_vec = TextVectorization.from_config(info['config'])
        mhc_vec.set_weights(info['weights'])
    else:
        raise Exception('MHC Vectorizer not found')
    if os.path.exists('./data/pep_vec.pkl'):
        with open('./data/pep_vec.pkl', 'rb') as f:
            info = pickle.load(f)
        pep_vec = TextVectorization.from_config(info['config'])
        pep_vec.set_weights(info['weights'])
    else:
        raise Exception('Peptide Vectorizer not found')
    model_path = path_formatter(args, f'{args.model}', 'h5')
    model = keras.models.load_model(model_path,
                                    custom_objects={'Attention': Attention})
    
    data_gen = DataGenerator(data, mhc_vec, pep_vec, args.batch_size, shuffle=False)
    model.evaluate(data_gen, batch_size=args.batch_size)

