import tensorflow as tf

flags = tf.flags
cfg = flags.FLAGS


# command-line config
flags.DEFINE_string ("data_path",  "data_ptb",          "Data path")
flags.DEFINE_string ("save_file",  "models/recent.dat", "Save file")
flags.DEFINE_string ("load_file",  "",                  "File to load model from")
flags.DEFINE_string ("vocab_file", "wvocab.pk",         "Word vocab pickle file in data "
                                                        "path")

flags.DEFINE_bool   ("preallocate_gpu",   False,   "Preallocate all of the GPU memory")
flags.DEFINE_integer("batch_size",        50,      "Batch size")
flags.DEFINE_integer("history_size",      5,       "n for the n-gram model")
flags.DEFINE_integer("emb_size",          224,     "Word embedding size")
flags.DEFINE_integer("hidden_size",       800,     "Hidden state size")
flags.DEFINE_integer("softmax_samples",   1024,    "Number of classes to sample for softmax")
flags.DEFINE_integer("max_sent_length",   256,     "Maximum length of a sentence in words")
flags.DEFINE_bool   ("training",          True,    "Training mode, turn off for testing")
flags.DEFINE_string ("optimizer",         "adam",  "Optimizer to use (sgd, adam, adagrad, "
                                                   "adadelta)")
flags.DEFINE_float  ("learning_rate",     1e-3,    "Optimizer initial learning rate")
flags.DEFINE_integer("max_epoch",         10000,   "Maximum number of epochs to run for")
flags.DEFINE_integer("max_steps",         9999999, "Maximum number of steps to run for")
flags.DEFINE_integer("gen_samples",       1,       "Number of demo samples batches to generate "
                                                   "per epoch")
flags.DEFINE_integer("gen_every",         500,     "Generate samples every these many training "
                                                   "steps (0 to disable, -1 for each epoch)")
flags.DEFINE_integer("print_every",       50,      "Print every these many steps")
flags.DEFINE_integer("save_every",        -1,      "Save every these many steps (0 to disable, "
                                                   "-1 for each epoch)")
flags.DEFINE_bool   ("save_overwrite",    True,    "Overwrite the same file each time")
flags.DEFINE_integer("validate_every",    1,       "Validate every these many epochs "
                                                   "(0 to disable)")


print('Config:')
cfg._parse_flags()
cfg_dict = cfg.__dict__['__flags']
maxlen = max(len(k) for k in cfg_dict)
for k, v in sorted(cfg_dict.items(), key=lambda x: x[0]):
    print(k.ljust(maxlen + 2), v)
print()
