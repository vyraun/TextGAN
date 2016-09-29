import tensorflow as tf

import rnncell
import utils


class EncoderDecoderModel(object):
    '''The encoder-decoder model.'''

    def __init__(self, config, vocab, training):
        self.config = config
        self.vocab = vocab
        self.training = training
        # left-aligned data:  <sos> w1 w2 ... w_T <eos> <pad...>
        self.ldata = tf.placeholder(tf.int32, [config.batch_size, None], name='ldata')
        # right-aligned data: <pad...> <sos> w1 s2 ... w_T
        self.rdata = tf.placeholder(tf.int32, [config.batch_size, None], name='rdata')

        sent_length = tf.shape(self.ldata)[1]
        lembs_dropped = self.word_embeddings(self.word_dropout(self.ldata))
        rembs = self.word_embeddings(self.rdata, reuse=True)
        state = self.encoder(rembs)
        latent = utils.highway(state)
        outputs = self.decoder(lembs_dropped, latent)
        # shift left the input to get the targets
        targets = tf.concat(1, [self.ldata[:,1:], tf.zeros([config.batch_size, 1], tf.int32)])
        loss = self.mle_loss(outputs, targets)
        self.nll = tf.reduce_sum(loss) / config.batch_size
        self.cost = self.nll
        if training:
            self.train_op = self.train(self.cost)
        else:
            self.train_op = tf.no_op()

    def rnn_cell(self, latent=None):
        '''Return a multi-layer RNN cell.'''
        return tf.nn.rnn_cell.MultiRNNCell([rnncell.GRUCell(self.config.hidden_size, latent=latent)
                                               for _ in xrange(self.config.num_layers)],
                                           state_is_tuple=True)

    def word_dropout(self, inputs):
        '''Randomly replace words from inputs with <unk>.'''
        if self.training and self.config.decoder_dropout > 0.0:
            unks = tf.ones_like(inputs, tf.int32) * self.vocab.unk_index
            mask = tf.cast(tf.greater(tf.nn.dropout(tf.cast(unks, tf.float32),
                                                    self.config.decoder_dropout), 0.0), tf.int32)
            return ((1 - mask) * inputs) + (mask * unks)
        else:
            return inputs

    def word_embeddings(self, inputs, reuse=False):
        '''Look up word embeddings for the input indices.'''
        with tf.device('/cpu:0') and tf.variable_scope("Embeddings", reuse=reuse):
            embedding = tf.get_variable('word_embedding', [len(self.vocab.vocab),
                                                           self.config.word_emb_size],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))
            embeds = tf.nn.embedding_lookup(embedding, inputs, name='word_embedding_lookup')
        return embeds

    def encoder(self, inputs):
        '''Encode sentence and return a latent representation.'''
        with tf.variable_scope("Encoder"):
            _, latent = tf.nn.dynamic_rnn(self.rnn_cell(), inputs, dtype=tf.float32)
        return latent

    def decoder(self, inputs, latent):
        '''Use the latent representation and word inputs to predict next words.'''
        if self.config.force_nolatent:
            latent = None
        if self.config.force_noinputs:
            inputs = tf.zeros_like(inputs)
        with tf.variable_scope("Decoder"):
            outputs, _ = tf.nn.dynamic_rnn(self.rnn_cell(latent), inputs, dtype=tf.float32)
        return outputs

    def mle_loss(self, outputs, targets):
        '''Maximum likelihood estimation loss.'''
        mask = tf.cast(tf.greater(targets, 0, name='targets_mask'), tf.float32)
        output = tf.reshape(tf.concat(1, outputs), [-1, self.config.hidden_size])
        softmax_w = tf.get_variable("softmax_w", [len(self.vocab.vocab), self.config.hidden_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
        softmax_b = tf.get_variable("softmax_b", [len(self.vocab.vocab)],
                                    initializer=tf.zeros_initializer)
        if self.training and self.config.softmax_samples < len(self.vocab.vocab):
            targets = tf.reshape(targets, [-1, 1])
            mask = tf.reshape(mask, [-1])
            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, output, targets,
                                              self.config.softmax_samples, len(self.vocab.vocab))
            loss *= mask
        else:
            logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(softmax_w),
                                              name='softmax_transform'), softmax_b)
            loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                          [tf.reshape(targets, [-1])],
                                                          [tf.reshape(mask, [-1])])
        return tf.reshape(loss, [self.config.batch_size, -1])

    def train(self, cost):
        '''Training op.'''
        self.lr = tf.Variable(0.0, trainable=False)
        if self.config.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.config.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.lr)
        elif self.config.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        if self.config.max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
        return optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        '''Change the learning rate'''
        print 'Setting learning rate to', lr_value
        session.run(tf.assign(self.lr, lr_value))
