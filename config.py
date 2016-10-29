import collections

import tensorflow as tf

flags = tf.flags

# command-line config
flags.DEFINE_string ("data_path",  "data",              "Data path")
flags.DEFINE_string ("save_file",  "models/recent.dat", "Save file")
flags.DEFINE_string ("load_file",  "",                  "File to load model from")
flags.DEFINE_string ("vocab_file", "data/vocab.pk",   "Vocab pickle file")

flags.DEFINE_integer("batch_size",        32,      "Batch size")
flags.DEFINE_integer("word_emb_size",     224,     "Number of learnable dimensions in word " \
                                                   "embeddings")
flags.DEFINE_integer("num_layers",        2,       "Number of RNN layers")
flags.DEFINE_integer("hidden_size",       192,     "RNN hidden state size")
flags.DEFINE_float  ("word_dropout",      0.33,    "Word dropout probability")
flags.DEFINE_integer("softmax_samples",   1000,    "Number of classes to sample for softmax")
flags.DEFINE_integer("generator_top_k",   1,       "Number of words to consider from previous " \
                                                   "timestep during generation (-1 for all)")
flags.DEFINE_float  ("maintain_d_acc",    0.95,    "Try to maintain descriminator accuracy to " \
                                                   "this value (accross prints)")
flags.DEFINE_float  ("d_acc_slack",       0.03,    "Let descriminator accuracy vary by this much")
flags.DEFINE_float  ("max_perplexity",    10.0,    "Scheduler maintains perplexity to be under " \
                                                   "this")
flags.DEFINE_integer("gen_sent_length",   96,      "Maximum length of a generated sentence")
flags.DEFINE_float  ("max_grad_norm",     5.0,     "Gradient clipping")
flags.DEFINE_bool   ("training",          True,    "Training mode, turn off for testing")
flags.DEFINE_integer("gan_wait_epochs",   1,       "The number of epochs to wait before " \
                                                   "optimizing the GAN objectives")
flags.DEFINE_string ("optimizer",         "adam",  "Optimizer to use (sgd, adam, adagrad, " \
                                                   "adadelta)")
flags.DEFINE_float  ("mle_learning_rate", 1e-3,    "Optimizer initial learning rate for MLE")
flags.DEFINE_float  ("d_learning_rate",   1e-3,    "Optimizer initial learning rate for " \
                                                   "discriminator")
flags.DEFINE_float  ("g_learning_rate",   1e-3,    "Optimizer initial learning rate for generator")
flags.DEFINE_integer("max_epoch",         50,      "Maximum number of epochs to run for")
flags.DEFINE_integer("max_steps",         9999999, "Maximum number of steps to run for")

flags.DEFINE_integer("gen_samples",       1,       "Number of demo samples batches to generate " \
                                                   "per epoch")
flags.DEFINE_integer("gen_every",         2500,    "Generate samples every these many training " \
                                                   "steps (0 to disable, -1 for each epoch)")
flags.DEFINE_integer("print_every",       50,      "Print every these many steps")
flags.DEFINE_integer("save_every",        -1,      "Save every these many steps (0 to disable, " \
                                                   "-1 for each epoch)")
flags.DEFINE_bool   ("save_overwrite",    True,    "Overwrite the same file each time")
flags.DEFINE_integer("validate_every",    1,       "Validate every these many epochs " \
                                                   "(0 to disable)")

# sanity check options
flags.DEFINE_bool   ("force_nolatent",    False,   "Force no latent input for decoder")
flags.DEFINE_bool   ("force_noinputs",    False,   "Force no true inputs for decoder")


class Config(object):
    def __init__(self):
        # copy flag values to attributes of this Config object
        for k, v in sorted(flags.FLAGS.__dict__['__flags'].items(), key=lambda x: x[0]):
            setattr(self, k, v)

