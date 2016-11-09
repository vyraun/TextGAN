import collections

import tensorflow as tf

flags = tf.flags
cfg = flags.FLAGS


# command-line config
flags.DEFINE_string ("data_path",       "data_short",           "Data path")
flags.DEFINE_string ("save_file",       "models/recent.dat",    "Save file")
flags.DEFINE_string ("load_file",       "",                     "File to load model from")
flags.DEFINE_string ("word_vocab_file", "data_short/wvocab.pk", "Word vocab pickle file")
flags.DEFINE_string ("char_vocab_file", "data_short/cvocab.pk", "Character vocab pickle file")

flags.DEFINE_bool   ("char_model",        True,    "Character-level model")
flags.DEFINE_integer("batch_size",        32,      "Batch size")
flags.DEFINE_integer("word_emb_size",     224,     "Word embedding size")
flags.DEFINE_integer("char_emb_size",     50,      "Character embedding size")
flags.DEFINE_integer("num_layers",        1,       "Number of RNN layers")
flags.DEFINE_integer("word_hidden_size",  256,     "RNN hidden state size for word model")
flags.DEFINE_integer("char_hidden_size",  1024,    "RNN hidden state size for char model")
flags.DEFINE_float  ("word_dropout",      0.0,     "Word dropout probability")
flags.DEFINE_integer("softmax_samples",   1000,    "Number of classes to sample for softmax")
flags.DEFINE_integer("generator_top_k",   1,       "Number of words to consider from previous "
                                                   "timestep during generation (-1 for all)")
flags.DEFINE_bool   ("encoder_after_gan", True,    "Update the encoder after GAN generator update")
flags.DEFINE_float  ("min_d_acc",         0.75,    "Update generator if descriminator is better "
                                                   "than this")
flags.DEFINE_float  ("max_d_acc",         0.99,    "Update descriminator if accuracy less than "
                                                   "this")
flags.DEFINE_float  ("max_perplexity",    -1,      "Scheduler maintains perplexity to be under "
                                                   "this (-1 to disable)")
flags.DEFINE_integer("sc_list_size",      6,       "Number of previous prints to look at in "
                                                   "scheduler")
flags.DEFINE_float  ("sc_decay",          0.6,     "Scheduler importance decay")
flags.DEFINE_bool   ("d_rnn",             True,    "Recurrent discriminator")
flags.DEFINE_integer("d_num_layers",      1,       "Number of RNN layers for discriminator (if "
                                                   "recurrent)")
flags.DEFINE_bool   ("d_rnn_bidirect",    True,    "Recurrent discriminator is bidirectional")
flags.DEFINE_integer("word_sent_length",  50,      "Maximum length of a generated sentence for "
                                                   "word model")
flags.DEFINE_integer("char_sent_length",  300,     "Maximum length of a generated sentence for "
                                                   "char model")
flags.DEFINE_float  ("max_grad_norm",     20.0,    "Gradient clipping")
flags.DEFINE_bool   ("training",          True,    "Training mode, turn off for testing")
flags.DEFINE_string ("mle_optimizer",     "adam",  "MLE optimizer to use (sgd, adam, adagrad, "
                                                   "adadelta)")
flags.DEFINE_string ("d_optimizer",       "adam",  "Discriminator optimizer to use (sgd, adam, "
                                                   "adagrad, adadelta)")
flags.DEFINE_string ("g_optimizer",       "adam",  "Generator optimizer to use (sgd, adam, "
                                                   "adagrad, adadelta)")
flags.DEFINE_float  ("mle_learning_rate", 1e-4,    "Optimizer initial learning rate for MLE")
flags.DEFINE_float  ("d_learning_rate",   1e-4,    "Optimizer initial learning rate for "
                                                   "discriminator")
flags.DEFINE_float  ("g_learning_rate",   1e-4,    "Optimizer initial learning rate for generator")
flags.DEFINE_integer("max_epoch",         50,      "Maximum number of epochs to run for")
flags.DEFINE_integer("max_steps",         9999999, "Maximum number of steps to run for")

flags.DEFINE_integer("gen_samples",       1,       "Number of demo samples batches to generate "
                                                   "per epoch")
flags.DEFINE_integer("gen_every",         2500,    "Generate samples every these many training "
                                                   "steps (0 to disable, -1 for each epoch)")
flags.DEFINE_integer("print_every",       50,      "Print every these many steps")
flags.DEFINE_integer("save_every",        -1,      "Save every these many steps (0 to disable, "
                                                   "-1 for each epoch)")
flags.DEFINE_bool   ("save_overwrite",    True,    "Overwrite the same file each time")
flags.DEFINE_integer("validate_every",    1,       "Validate every these many epochs "
                                                   "(0 to disable)")


if cfg.char_model:
    cfg.emb_size = cfg.char_emb_size
    cfg.hidden_size = cfg.char_hidden_size
    cfg.gen_sent_length = cfg.char_sent_length
else:
    cfg.emb_size = cfg.word_emb_size
    cfg.hidden_size = cfg.word_hidden_size
    cfg.gen_sent_length = cfg.word_sent_length
