import itertools
import re

import nltk


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def print_color(s, color=None):
    if color:
        print color + str(s) + Colors.ENDC,
    else:
        print s,


fix_re = re.compile(r'''[^a-z0-9"'?.,]+''')
num_re = re.compile(r'[0-9]+')

def fix_word(word):
    word = word.lower()
    word = fix_re.sub('', word)
    word = num_re.sub('#', word)
    return word


def read_words(line):
    for raw_word in nltk.word_tokenize(line.replace('<unk>', '-unk-')): # workaround to get the NLTK tokenization deal with <unk> nicely
        if raw_word == '-unk-':
            yield '<unk>'
        else:
            word = fix_word(raw_word)
            if word:
                yield word


def grouper(n, iterable, fillvalue=None):
    '''Group elements of iterable in groups of n. For example:
       >>> [e for e in grouper(3, [1,2,3,4,5,6,7])]
       [(1, 2, 3), (4, 5, 6), (7, None, None)]'''
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)
