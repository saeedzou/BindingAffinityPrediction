import os
import argparse

import numpy as np
import pandas as pd
from gensim.models import word2vec
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def split_ngrams(seq, n):
    return [list("".join(ngram) for ngram in zip(*[iter(seq[i:])]*n)) for i in range(n)]


def generate_corpusfile(corpus_fname, n, out, col='MHC_sequence'):
    df = pd.read_csv(corpus_fname, index_col=0)
    with open(out, "w") as f:
        for r in tqdm(df[col]):
            ngram_patterns = split_ngrams(r, n)
            for ngram_pattern in ngram_patterns:
                f.write(" ".join(ngram_pattern) + "\n")	# Take all the sequences and split them into kmers


class ProtVec(word2vec.Word2Vec):
    def __init__(self, corpus_fname=None, n=3, size=100, out="output_corpus.txt", sg=1, window=25, min_count=1, workers=9, col='MHC_sequence'):
        self.size = size
        self.corpus_fname = corpus_fname
        self.sg = sg
        self.window = window
        self.workers = workers
        self.out = out
        self.vocab = min_count
        self.col = col

        if(corpus_fname is not None):
            if(not os.path.isfile(out)):
                print("-- Generating corpus --")
                generate_corpusfile(corpus_fname, n, out, col)
            else:
                print("-- Corpus File Found --")
		
        self.corpus = word2vec.Text8Corpus(out)
        print("-- Corpus Setup Successful --")

    def word2vec_init(self, vectors_txt, model_weights):
        print("-- Initializing Word2Vec model --")
        print("-- Training the model --")
        self.m = word2vec.Word2Vec(self.corpus, vector_size=self.size, sg=self.sg, window=self.window, min_count=self.vocab, workers=self.workers)
        self.m.wv.save_word2vec_format(vectors_txt)
        self.m.save(model_weights)
        print("-- Saving Model Weights to : %s " % (vectors_txt))

    def load_protvec(self, model_weights):
        print("-- Load Word2Vec model --")
        self.m = word2vec.Word2Vec.load(model_weights)
        return self.m


def tsne_plot(model, n_components=2, random_state=42):
	"""
	Create a TSNE model and plot it
	"""
	print("-- Start t-SNE plot --")
	labels = []
	tokens = []

	for word in model.wv.vocab:
		tokens.append(model[word])
		labels.append(word)

	tsne_model = TSNE(n_components=n_components, random_state=random_state)
	new_values = tsne_model.fit_transform(tokens)

	x = []
	y = []
	for value in new_values:
		x.append(value[0])
		y.append(value[1])
	
	plt.figure(figsize=(16, 16))
	for i in range(len(x)):
		plt.scatter(x[i], y[i])
		plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")
	plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ProtVec using gensim (Word2Vec)")
    parser.add_argument("-f", "--corpus_fname", default="./data/binding_affinity_data.csv", help="path to the input csv file")
    parser.add_argument("-c", "--col", default="MHC_sequence", help="column name of the sequences")
    parser.add_argument("-o", "--output_corpus", default="./data/output_corpus_3.txt", help="path to the output corpus")
    parser.add_argument("-n", "--ngram_length", type=int, default=3, help="ngram length")
    parser.add_argument("--skip_gram", type=int, default=1, help="to enable skip-gram algorithm")
    parser.add_argument("--window", type=int, default=25, help="set window size")
    parser.add_argument("--min_count", type=int, default=1, help="neglect those words whose frequency is less than this threshold")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("-s", "--size", type=int, default=100, help="embedding dimension")
    parser.add_argument("-v", "--vectors", default="./data/3-gram-vectors.txt", help="path to the text file where the vectors are to be stored")
    parser.add_argument("-m", "--model_weights", default="./data/3-gram-model-weights.mdl", help="path to the binary file where the model weights are to be stored")
    args = parser.parse_args()

    model = ProtVec(corpus_fname=args.corpus_fname, 
                    n=args.ngram_length, 
                    out=args.output_corpus, 
                    sg=args.skip_gram, 
                    window=args.window, 
                    min_count=args.min_count, 
                    workers=args.workers,
                    col=args.col)
    # check if args.vectors exist
    if(not os.path.isfile(args.vectors)):
        model.word2vec_init(args.vectors, args.model_weights)
    else:
        model.load_protvec(args.model_weights)

    tsne_plot(model.m, 3)
    
    if args.ngram_length == 3:
         w1 = 'WFN'
         print("Most similar to %s:" % w1, model.m.wv.most_similar(positive=w1))

         w2 = 'WYF'
         print("Most similar to %s:" % w2, model.m.wv.most_similar(positive=w2))

