import collections

import tensorflow as tf

flags = tf.flags

# command-line config
flags.DEFINE_string ("data_path",  "data",              "Data path")
flags.DEFINE_string ("save_file",  "models/recent.dat", "Save file")
flags.DEFINE_string ("load_file",  "",                  "File to load model from")
flags.DEFINE_string ("vocab_file", "models/vocab.pk",   "Vocab pickle file")

flags.DEFINE_integer("batch_size",    32,   "Batch size")
flags.DEFINE_integer("word_emb_size", 256,  "Number of learnable dimensions in word embeddings")
flags.DEFINE_integer("num_layers",    2,    "Number of RNN layers")
flags.DEFINE_integer("hidden_size",   192,  "RNN hidden state size")
flags.DEFINE_bool   ("training",      True, "Training mode, turn off for testing")


class Config(object):
    def __init__(self):
        # copy flag values to attributes of this Config object
        for k, v in sorted(flags.FLAGS.__dict__['__flags'].items(), key=lambda x: x[0]):
            setattr(self, k, v)

