from pathlib import Path
import pickle
import random

import numpy as np
import tensorflow as tf

from config import cfg
import utils


class Vocab(object):

    '''Stores the vocab: forward and reverse mappings'''

    def __init__(self):
        self.vocab = ['<sos>', '<unk>']
        self.vocab_lookup = {w: i for i, w in enumerate(self.vocab)}
        self.sos_index = self.vocab_lookup.get('<sos>')
        self.unk_index = self.vocab_lookup.get('<unk>')

    def load_by_parsing(self, save=False, verbose=True):
        '''Read the vocab from the dataset'''
        if verbose:
            print('Loading vocabulary by parsing...')
        fnames = Path(cfg.data_path).glob('*.txt')
        for fname in fnames:
            if verbose:
                print(fname)
            with fname.open('r') as f:
                for line in f:
                    for word in utils.read_words(line, chars=cfg.char_model):
                        if word not in self.vocab_lookup:
                            self.vocab_lookup[word] = len(self.vocab)
                            self.vocab.append(word)
        if verbose:
            print('Vocabulary loaded, size:', len(self.vocab))

    def load_from_pickle(self, verbose=True):
        '''Read the vocab from a pickled file'''
        if cfg.char_model:
            pkfile = cfg.char_vocab_file
        else:
            pkfile = cfg.word_vocab_file
        try:
            if verbose:
                print('Loading vocabulary from pickle...')
            with open(pkfile, 'rb') as f:
                self.vocab, self.vocab_lookup = pickle.load(f)
            if verbose:
                print('Vocabulary loaded, size:', len(self.vocab))
        except IOError:
            if verbose:
                print('Error loading from pickle, attempting parsing.')
            self.load_by_parsing(save=True, verbose=verbose)
            with open(pkfile, 'wb') as f:
                pickle.dump([self.vocab, self.vocab_lookup], f, -1)
                if verbose:
                    print('Saved pickle file.')

    def lookup(self, words):
        return [self.vocab_lookup.get(w) for w in words]


class Reader(object):
    def __init__(self, vocab):
        self.vocab = vocab
        random.seed(0)  # deterministic random

    def read_lines(self, fnames):
        '''Read single lines from data'''
        for fname in fnames:
            with fname.open('r') as f:
                for line in f:
                    yield self.vocab.lookup([w for w in utils.read_words(line,
                                                                         chars=cfg.char_model)])

    def _prepare(self, lines):
        '''Prepare non-overlapping data'''
        seqs = []
        seq = []
        for line in lines:
            line.insert(0, self.vocab.sos_index)
            for word in line:
                seq.append(word)
                if len(seq) == cfg.max_sent_length:
                    seqs.append(seq)
                    seq = []
        return seqs

    def buffered_read(self, fnames, buffer_size=500):
        '''Read and yield a list of non-overlapping sequences'''
        buffer_size = max(buffer_size, cfg.max_sent_length)
        lines = []
        for line in self.read_lines(fnames):
            lines.append(line)
            if len(lines) == buffer_size:
                random.shuffle(lines)
                yield self._prepare(lines)
                lines = []
        if lines:
            random.shuffle(lines)
            yield self._prepare(lines)

    def buffered_read_batches(self, fnames, buffer_size=500):
        batches = []
        batch = []
        for lines in self.buffered_read(fnames):
            for line in lines:
                batch.append(line)
                if len(batch) == cfg.batch_size:
                    batches.append(self.pack(batch))
                    if len(batches) == buffer_size:
                        random.shuffle(batches)
                        for batch in batches:
                            yield batch
                        batches = []
                    batch = []
        # ignore current incomplete batch
        if batches:
            random.shuffle(batches)
            for batch in batches:
                yield batch

    def pack(self, batch):
        '''Pack python-list batches into numpy batches'''
        ret_batch = np.zeros([cfg.batch_size, cfg.max_sent_length], dtype=np.int32)
        for i, s in enumerate(batch):
            ret_batch[i, :len(s)] = s
        return ret_batch

    def training(self):
        '''Read batches from training data'''
        yield from self.buffered_read_batches([Path(cfg.data_path) / 'train.txt'])

    def validation(self):
        '''Read batches from validation data'''
        yield from self.buffered_read_batches([Path(cfg.data_path) / 'valid.txt'])

    def testing(self):
        '''Read batches from testing data'''
        yield from self.buffered_read_batches([Path(cfg.data_path) / 'test.txt'])


def main(_):
    '''Reader tests'''

    vocab = Vocab()
    vocab.load_from_pickle()

    reader = Reader(vocab)
    c = 0
    w = 0
    for batch in reader.training():
        n_words = np.sum(batch != 0)
        w += n_words
        c += len(batch)
        for line in batch:
            print(line)
            for e in line:
                if cfg.char_model:
                    print(vocab.vocab[e], end='')
                else:
                    print(vocab.vocab[e], end=' ')
            print()
            print()
    print('Total lines:', c)
    print('Total words:', w)


if __name__ == '__main__':
    tf.app.run()
