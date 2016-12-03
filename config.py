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
flags.DEFINE_bool   ("use_gan",           True,    "Use adversarial objectives")
flags.DEFINE_bool   ("enc_bidirect",      True,    "Use a bidirectional encoder")
flags.DEFINE_integer("batch_size",        32,      "Batch size")
flags.DEFINE_integer("word_emb_size",     224,     "Word embedding size")
flags.DEFINE_integer("char_emb_size",     96,      "Character embedding size")
flags.DEFINE_integer("num_layers",        1,       "Number of RNN layers")
flags.DEFINE_integer("beam_size",         5,       "Beam size for beam search")
flags.DEFINE_integer("word_hidden_size",  512,     "RNN hidden state size for word model")
flags.DEFINE_integer("char_hidden_size",  768,     "RNN hidden state size for char model")
flags.DEFINE_integer("latent_size",       48,      "Latent representation size")
flags.DEFINE_float  ("word_dropout",      0.66,    "Word (or char) dropout probability for decoder "
                                                   "inputs")
flags.DEFINE_integer("softmax_samples",   1024,    "Number of classes to sample for softmax")
flags.DEFINE_bool   ("concat_inputs",     True,    "Concatenate inputs to states before "
                                                   "discriminating")
flags.DEFINE_integer("generator_top_k",   1,       "Number of words to consider from previous "
                                                   "timestep during generation (-1 for all)")
flags.DEFINE_bool   ("variational",       True,    "Variational encoder-decoder")
flags.DEFINE_bool   ("generate_argmax",   False,   "Use argmax instead of sampling during "
                                                   "generation")
flags.DEFINE_bool   ("encoder_after_gan", True,    "Update the encoder after GAN generator update")
flags.DEFINE_float  ("min_d_acc",         0.75,    "Update generator if descriminator is better "
                                                   "than this")
flags.DEFINE_float  ("max_d_acc",         0.95,    "Update descriminator if accuracy less than "
                                                   "this")
flags.DEFINE_float  ("max_perplexity",    -1,      "Scheduler maintains perplexity to be under "
                                                   "this (-1 to disable)")
flags.DEFINE_bool   ("sc_use_kld_weight", False,   "Scheduler waits for KL divergence weight to "
                                                   "reach its max")
flags.DEFINE_integer("sc_list_size",      3,       "Number of previous prints to look at in "
                                                   "scheduler")
flags.DEFINE_float  ("sc_decay",          0.1,     "Scheduler importance decay")
flags.DEFINE_bool   ("d_rnn",             True,    "Recurrent discriminator")
flags.DEFINE_bool   ("d_energy_based",    False,   "Energy-based discriminator")
flags.DEFINE_float  ("d_word_eb_margin",  512.0,   "Margin for energy-based discriminator for word "
                                                   "model")
flags.DEFINE_float  ("d_char_eb_margin",  1024.0,  "Margin for energy-based discriminator for char "
                                                   "model")
flags.DEFINE_integer("d_num_layers",      1,       "Number of RNN layers for discriminator (if "
                                                   "recurrent)")
flags.DEFINE_bool   ("d_rnn_bidirect",    True,    "Recurrent discriminator is bidirectional")
flags.DEFINE_integer("d_conv_window",     5,       "Convolution window for convolution on "
                                                   "discriminative RNN's states")
flags.DEFINE_integer("word_sent_length",  192,     "Maximum length of a sentence for word model")
flags.DEFINE_integer("char_sent_length",  480,     "Maximum length of a sentence for char model")
flags.DEFINE_float  ("max_grad_norm",     20.0,    "Gradient clipping")
flags.DEFINE_integer("anneal_bias",       1000,    "The step to reach half-point for KL divergence "
                                                   "weight annealing")
flags.DEFINE_float  ("anneal_max",        0.1,     "The maximum KL divergence weight")
flags.DEFINE_bool   ("training",          True,    "Training mode, turn off for testing")
flags.DEFINE_string ("d_optimizer",       "adam",  "Discriminator optimizer to use (sgd, adam, "
                                                   "adagrad, adadelta)")
flags.DEFINE_string ("g_optimizer",       "adam",  "Generator optimizer to use (sgd, adam, "
                                                   "adagrad, adadelta)")
flags.DEFINE_float  ("d_learning_rate",   1e-4,    "Optimizer initial learning rate for "
                                                   "discriminator")
flags.DEFINE_float  ("g_learning_rate",   1e-4,    "Optimizer initial learning rate for generator")
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


if cfg.char_model:
    cfg.emb_size = cfg.char_emb_size
    cfg.hidden_size = cfg.char_hidden_size
    cfg.max_sent_length = cfg.char_sent_length
    cfg.d_eb_margin = cfg.d_char_eb_margin
else:
    cfg.emb_size = cfg.word_emb_size
    cfg.hidden_size = cfg.word_hidden_size
    cfg.max_sent_length = cfg.word_sent_length
    cfg.d_eb_margin = cfg.d_word_eb_margin
