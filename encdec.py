import tensorflow as tf

import rnncell
import utils


class EncoderDecoderModel(object):
    '''The encoder-decoder model.'''

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab
        # left-aligned data:  <sos> w1 w2 ... w_T <eos> <pad...>
        ldata = tf.placeholder(tf.int32, [config.batch_size, None], name='ldata')
        # right-aligned data: <pad...> <sos> w1 s2 ... w_T
        rdata = tf.placeholder(tf.int32, [config.batch_size, None], name='rdata')
        # masks where padding words are 0 and all others are 1
        ldata_mask = tf.greater(ldata, 0, name='ldata_mask')
        rdata_mask = tf.greater(rdata, 0, name='rdata_mask')

        sent_length = tf.shape(ldata)[1]
        lembs = self.word_embeddings(ldata)
        rembs = self.word_embeddings(rdata, reuse=True)
        state = self.encoder(rembs)
        latent = utils.highway(state)
        outputs = self.decoder(lembs, latent)
        loss = self.mle_loss(outputs, ldata[:,1:], ldata_mask[:,1:])
        self.cost = tf.reduce_sum(loss) / config.batch_size
        if config.training:
            self.train_op = self.train(self.cost)
        else:
            self.train_op = tf.no_op()

    def rnn_cell(self, latent=None):
        '''Return a multi-layer RNN cell.'''
        return tf.nn.rnn_cell.MultiRNNCell([rnncell.GRUCell(self.config.hidden_size, latent=latent)
                                               for _ in xrange(self.config.num_layers)],
                                           state_is_tuple=True)

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
        with tf.variable_scope("Decoder"):
            outputs, _ = tf.nn.dynamic_rnn(self.rnn_cell(latent), inputs, dtype=tf.float32)
        return outputs

    def mle_loss(self, outputs, targets, mask):
        '''Maximum likelihood estimation loss.'''
        output = tf.reshape(tf.concat(1, outputs), [-1, self.config.hidden_size])
        softmax_w = tf.get_variable("softmax_w", [len(self.vocab.vocab), self.config.hidden_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
        softmax_b = tf.get_variable("softmax_b", [len(self.vocab.vocab)],
                                    initializer=tf.zeros_initializer)
        if self.config.training and self.config.softmax_samples < len(self.vocab.vocab):
            targets = tf.reshape(targets, [-1, 1])
            mask = tf.reshape(mask, [-1])
            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, output, targets,
                                          self.config.softmax_samples, len(self.vocab.vocab)) * mask
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
        if config.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'adadelta':
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
