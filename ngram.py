import tensorflow as tf

from config import cfg
import utils


class NGramModel(object):

    '''The neural n-gram language model.'''

    def __init__(self, vocab, training, optimizer=None):
        self.vocab = vocab
        self.training = training
        self.optimizer = optimizer

        self.embedding = self.word_embedding_matrix()
        self.softmax_w, self.softmax_b = self.softmax_variables()

        with tf.variable_scope("GlobalMLE"):
            self.global_step = tf.get_variable('global_step', shape=[],
                                               initializer=tf.zeros_initializer,
                                               trainable=False)
        # input data
        self.data = tf.placeholder(tf.int32, [cfg.batch_size, cfg.history_size + 1], name='data')

        embs = self.word_embeddings(self.data[:, :cfg.history_size])
        targets = self.data[:, -1]
        output = self.predict(embs)

        self.nll = tf.reduce_sum(self.mle_loss(output, targets)) / cfg.batch_size
        self.mle_cost = self.nll
        if training:
            self.mle_train_op = self.train_mle(self.mle_cost)
        else:
            self.mle_train_op = tf.no_op()

    def word_embedding_matrix(self):
        '''Define the word embedding matrix.'''
        with tf.device('/cpu:0') and tf.variable_scope("Embeddings"):
            embedding = tf.get_variable('word_embedding', [len(self.vocab.vocab),
                                                           cfg.emb_size],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))
        return embedding

    def softmax_variables(self):
        '''Define the softmax weight and bias variables.'''
        with tf.variable_scope("MLE_Softmax"):
            softmax_w = tf.get_variable("W", [len(self.vocab.vocab), cfg.emb_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("b", [len(self.vocab.vocab)],
                                        initializer=tf.zeros_initializer)
        return softmax_w, softmax_b

    def word_embeddings(self, inputs):
        '''Look up word embeddings for the input indices.'''
        with tf.device('/cpu:0'):
            embeds = tf.nn.embedding_lookup(self.embedding, inputs, name='word_embedding_lookup')
        return embeds

    def predict(self, inputs):
        '''Use the word inputs to predict next word.'''
        with tf.variable_scope("Predictor"):
            inputs = tf.reshape(inputs, [-1, cfg.emb_size * cfg.history_size])
            l1 = tf.nn.elu(utils.linear(inputs, cfg.hidden_size, True, 0.0,
                                        scope='hidden_transform'))
            output = tf.nn.elu(utils.linear(l1, cfg.emb_size, True, 0.0, scope='output_transform'))
        return output

    def mle_loss(self, output, targets):
        '''Maximum likelihood estimation loss.'''
        # don't enfoce loss on true <unk>'s, makes the reported perlexity slightly overestimated
        mask = tf.cast(tf.not_equal(targets, self.vocab.unk_index, name='unk_mask'), tf.float32)
        if self.training and cfg.softmax_samples < len(self.vocab.vocab):
            targets = tf.reshape(targets, [-1, 1])
            mask = tf.reshape(mask, [-1])
            loss = tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, output, targets,
                                              cfg.softmax_samples, len(self.vocab.vocab))
        else:
            logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(self.softmax_w),
                                              name='softmax_transform_mle'), self.softmax_b)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
        return loss * mask

    def train_mle(self, cost):
        '''Generic training helper'''
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(cost, tvars)
        return self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
