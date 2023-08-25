import keras
from keras.layers import TextVectorization
import pandas as pd
import numpy as np
import logging, argparse, pickle, os
import tensorflow as tf
from utils import load_csv, path_formatter, adapt_vectorizer, display_history
from dataloader import DataGenerator
from models import mhcAttentionAdd, mhcAttentionCombined, mhcAttentionConcat, mhcAttentionSimple, mhcNoAttention

# Set the random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

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
    parser.add_argument('-m', '--model', type=str, default='mhcAttentionAdd', 
                        help='Model to use. Options: mhcAttentionAdd, mhcAttentionCombined, mhcAttentionConcat, mhcAttentionSimple, mhcNoAttention')
    args = parser.parse_args()
    log_path = path_formatter(args, 'train', 'log')
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
    # Define a dictionary mapping model names to classes
    model_classes = {
        'mhcAttentionAdd': mhcAttentionAdd,
        'mhcAttentionCombined': mhcAttentionCombined,
        'mhcAttentionConcat': mhcAttentionConcat,
        'mhcAttentionSimple': mhcAttentionSimple,
        'mhcNoAttention': mhcNoAttention
        }

    # Check if the provided model name is valid
    if args.model not in model_classes:
        raise ValueError(f'Invalid model name: {args.model}')

    # Instantiate the selected model class
    model_class = model_classes[args.model]
    model = model_class(
        embedding_dim=args.embedding_dim,
        rnn_units=args.rnn_units,
        seq_len=args.seq_len,
        context_dim=args.context_dim,
        vocab_size=args.vocab_size,
        fc_in_units=args.fc_in_units,
        fc_out_units=args.fc_out_units
    )

    logging.info('Model Summary')
    model.build(input_shape=[(None, 1), (None, 1)])
    keras.utils.plot_model(model, to_file=f'./data/{args.model}.png', show_shapes=True, dpi=512)
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
    model_path = path_formatter(args, 'model', 'h5')
    model.save(model_path)
    logging.info('Saving History')
    hist = pd.DataFrame(h.history)
    hist_path = path_formatter(args, 'history', 'csv')
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