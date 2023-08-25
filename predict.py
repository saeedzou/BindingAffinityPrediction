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
    parser.add_argument('-b', '--batch_size', type=int, default=2048, help='Batch size')
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
    
    model = keras.models.load_model('./log/{args.model}_e10_bs2048_lr0.001_lossbinary_crossentropy_ed256_ru128_sl100_cd16_vs22_fci64_fco64.h5',
                                    custom_objects={'Attention': Attention})
    
    data_gen = DataGenerator(data, mhc_vec, pep_vec, args.batch_size, shuffle=False)
    model.evaluate(data_gen, batch_size=2048)

