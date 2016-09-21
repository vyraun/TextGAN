from __future__ import division

import cPickle as pickle
import glob
from os.path import join as pjoin

import nltk
import numpy as np
import tensorflow as tf

from config import Config
import utils


class Vocab(object):
    '''Stores the vocab: forward and reverse mappings'''
    def __init__(self, config):
        self.config = config
        self.vocab = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab_lookup = {w:i for i,w in enumerate(self.vocab)}

    def load_by_parsing(self, save=False, verbose=True):
        '''Read the vocab from the dataset'''
        if verbose:
            print 'Loading vocabulary by parsing...'
        fnames = glob.glob(pjoin(self.config.data_path, '*.txt'))
        for fname in fnames:
            if verbose:
                print fname
            with open(fname, 'r') as f:
                for line in f:
                    for word in utils.read_words(line):
                        if word not in self.vocab_lookup:
                            self.vocab_lookup[word] = len(self.vocab)
                            self.vocab.append(word)
        if verbose:
            print 'Vocabulary loaded, size:', len(self.vocab)

    def load_from_pickle(self, verbose=True):
        '''Read the vocab from a pickled file'''
        pkfile = self.config.vocab_file
        try:
            if verbose:
                print 'Loading vocabulary from pickle...'
            with open(pkfile, 'rb') as f:
                self.vocab, self.vocab_lookup = pickle.load(f)
            if verbose:
                print 'Vocabulary loaded, size:', len(self.vocab)
        except IOError:
            if verbose:
                print 'Error loading from pickle, attempting parsing.'
            self.load_by_parsing(save=True, verbose=verbose)
            with open(pkfile, 'wb') as f:
                pickle.dump([self.vocab, self.vocab_lookup], f, -1)
                if verbose:
                    print 'Saved pickle file.'

    def lookup(self, words):
        unk_index = self.vocab_lookup.get('<unk>')
        sos_index = self.vocab_lookup.get('<sos>')
        eos_index = self.vocab_lookup.get('<eos>')
        return [sos_index] + [self.vocab_lookup.get(w, unk_index) for w in words] + [eos_index]


class Reader(object):
    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

    def read(self, fnames):
        for fname in fnames:
            with open(fname, 'r') as f:
                for line in f:
                    print line
                    yield self.vocab.lookup([w for w in utils.read_words(line)])

    def training(self):
        yield self.read([pjoin(self.config.data_path, 'train.txt')])

    def validation(self):
        yield self.read([pjoin(self.config.data_path, 'valid.txt')])

    def testing(self):
        yield self.read([pjoin(self.config.data_path, 'test.txt')])


def main(_):
    config = Config()

    vocab = Vocab(config)
    vocab.load_from_pickle()

    reader = Reader(config, vocab)
    for batch in reader.training():
        for line in batch:
            print line
            for e in line:
                print vocab.vocab[e],
            print
            print


if __name__ == '__main__':
    tf.app.run()
