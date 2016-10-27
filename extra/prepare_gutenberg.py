from __future__ import division

import codecs
import collections
import glob
from os.path import join as pjoin
import random
import re
import unicodedata

import nltk

input_dir = 'gutenberg' # raw text dir

data_coverage = 0.97 # decide vocab based on how much of the data should be covered

MIN_LEN = 4
MAX_LEN = 49 # 12 for data_short

val_split = 0.0004 # gutenberg is huge
test_split = 0.0006
train_split = 1.0 - val_split - test_split


fix_re = re.compile(r"[^a-z0-9]+")
num_re = re.compile(r'[0-9]+')


def fix_word(word):
    word = word.lower()
    word = fix_re.sub('', word)
    word = num_re.sub('#', word)
    if not any(c.isalpha() for c in word):
        word = ''
    return word


def process(output, vocab, lines):
    if not lines:
        return
    para = unicodedata.normalize('NFKC', ' '.join(lines))
    for sent in nltk.sent_tokenize(para):
        words = [fix_word(w) for w in nltk.word_tokenize(sent)]
        words = [w for w in words if w]
        for word in words:
            vocab[word] += 1
        if len(words) >= MIN_LEN and len(words) <= MAX_LEN: # ignore very short and long sentences
            output.append(words)


def create_file(fname, lines, vocab):
    with open(fname, 'w') as f:
        for line in lines:
            words = []
            for w in line:
                if w in vocab:
                    words.append(w)
                else:
                    words.append('<unk>')
            print >> f, ' '.join(words)


def summarize(output, vocab):
    print
    print 'Size of corpus:', vocab.N()
    print 'Total vocab size:', vocab.B()

    N = len(output)
    test_N = int(test_split * N)
    val_N = int(val_split * N)
    train_N = N - test_N - val_N
    print 'Number of lines:', N
    print '   Train:', train_N
    print '   Val:  ', val_N
    print '   Test: ', test_N
    print
    return train_N, val_N, test_N


if __name__ == '__main__':
    output = []
    vocab = nltk.FreqDist()
    print 'Reading...'
    fnames = sorted(glob.glob(pjoin(input_dir, '*.txt')))
    for i, fname in enumerate(fnames):
        print '(%d/%d) %s' % (i, len(fnames), fname)
        with codecs.open(fname, 'r', 'latin-1') as f:
            paragraph = []
            for l in f:
                line = l.strip()
                if not line:
                    process(output, vocab, paragraph)
                    paragraph = []
                else:
                    paragraph.append(line)
            process(output, vocab, paragraph)
        if (i+1) % 50 == 0:
            summarize(output, vocab)
    train_N, val_N, test_N = summarize(output, vocab)
    top_words = vocab.most_common()
    count = 0
    for vocab_size in xrange(vocab.B()):
        count += top_words[vocab_size][1]
        if count / vocab.N() >= data_coverage:
            top_words = set(w for w,c in vocab.most_common(vocab_size+1))
            break
    print 'Final vocab size:', len(top_words)

    random.shuffle(output)
    create_file('train.txt', output[:train_N], top_words)
    create_file('test.txt', output[train_N:train_N+test_N], top_words)
    create_file('valid.txt', output[train_N+test_N:], top_words)
