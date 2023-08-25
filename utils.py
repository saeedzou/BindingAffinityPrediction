import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path):
    return pd.read_csv(path, index_col=0)

def path_formatter(args, type, format):
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

