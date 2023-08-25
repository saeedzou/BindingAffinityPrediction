import numpy as np
import keras
import tensorflow as tf
# Set the random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

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
