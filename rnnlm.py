import tensorflow as tf

from config import cfg
import rnncell
import utils


class RNNLMModel(object):

    '''The adversarial recurrent language model.'''

    def __init__(self, vocab, training, use_gan=True, g_optimizer=None, d_optimizer=None):
        self.vocab = vocab
        self.training = training
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.embedding = self.word_embedding_matrix()
        self.softmax_w, self.softmax_b = self.softmax_variables()

        with tf.variable_scope("GlobalMLE"):
            self.global_step = tf.get_variable('global_step', shape=[],
                                               initializer=tf.zeros_initializer,
                                               trainable=False)
        # input data
        self.data = tf.placeholder(tf.int32, [cfg.batch_size, cfg.max_sent_length], name='data')

        embs = self.word_embeddings(self.data)
        output, mle_states, _ = self.generator(embs, True)
        _, gan_states, self.generated = self.generator(embs, False, True)
        if use_gan:
            states = tf.concat(0, [mle_states, gan_states])

            if cfg.d_energy_based:
                d_out = self.discriminator_energy(states)
                d_loss, g_loss = self.gan_energy_loss(d_out[:, :-1, :], states)
                self.d_cost = tf.reduce_sum(d_loss) / (2 * cfg.batch_size)
                self.g_cost = tf.reduce_sum(g_loss) / (2 * cfg.batch_size)
            else:
                if cfg.d_rnn:
                    d_out = self.discriminator_rnn(states)
                else:
                    d_out = self.discriminator_finalstate(states)
                targets = tf.concat(0, [tf.ones([cfg.batch_size, 1]),
                                        tf.zeros([cfg.batch_size, 1])])
                gan_loss = self.gan_loss(d_out, targets)
                self.d_cost = tf.reduce_sum(gan_loss) / (2 * cfg.batch_size)
                self.g_cost = -self.d_cost
        else:
            self.d_cost = tf.zeros([])
            self.g_cost = tf.zeros([])

        # shift left the input to get the targets
        targets = tf.concat(1, [self.data[:, 1:], tf.zeros([cfg.batch_size, 1], tf.int32)])
        self.nll = tf.reduce_sum(self.mle_loss(output, targets)) / cfg.batch_size
        self.mle_cost = self.nll
        if training:
            self.mle_train_op = self.train_mle(self.mle_cost)
        else:
            self.mle_train_op = tf.no_op()
        if training and use_gan:
            self.d_train_op = self.train_d(self.d_cost)
            self.g_train_op = self.train_g(self.g_cost)
        else:
            self.d_train_op = tf.no_op()
            self.g_train_op = tf.no_op()

    def rnn_cell(self, num_layers, hidden_size, embedding=None, softmax_w=None, softmax_b=None,
                 return_states=False, pretanh=False):
        '''Return a multi-layer RNN cell.'''
        return rnncell.MultiRNNCell([rnncell.GRUCell(hidden_size, pretanh=pretanh)
                                     for _ in range(num_layers)], embedding=embedding,
                                    softmax_w=softmax_w, softmax_b=softmax_b,
                                    return_states=return_states, pretanh=pretanh)

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
            softmax_w = tf.get_variable("W", [len(self.vocab.vocab), cfg.hidden_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("b", [len(self.vocab.vocab)],
                                        initializer=tf.zeros_initializer)
        return softmax_w, softmax_b

    def word_embeddings(self, inputs):
        '''Look up word embeddings for the input indices.'''
        with tf.device('/cpu:0'):
            embeds = tf.nn.embedding_lookup(self.embedding, inputs, name='word_embedding_lookup')
        return embeds

    def generator(self, inputs, mle_mode, reuse=None):
        '''Use the word inputs to predict next words.'''
        with tf.variable_scope("Generator", reuse=reuse):
            if mle_mode:
                cell = self.rnn_cell(cfg.num_layers, cfg.hidden_size, return_states=True,
                                     pretanh=True)
            else:
                cell = self.rnn_cell(cfg.num_layers, cfg.hidden_size, self.embedding,
                                     self.softmax_w, self.softmax_b, return_states=True,
                                     pretanh=True)
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, swap_memory=True, dtype=tf.float32)
            output = tf.slice(outputs, [0, 0, 0], [-1, -1, cfg.hidden_size])
            if mle_mode:
                generated = None
                skip = 0
            else:
                generated = tf.squeeze(tf.cast(tf.slice(outputs, [0, 0, cfg.hidden_size],
                                                        [-1, -1, 1]),
                                               tf.int32), [-1])
                skip = 1
            states = tf.slice(outputs, [0, 0, cfg.hidden_size + skip], [-1, -1, -1])
        return output, states, generated

    def mle_loss(self, outputs, targets):
        '''Maximum likelihood estimation loss.'''
        present_mask = tf.greater(targets, 0, name='present_mask')
        # don't enfoce loss on true <unk>'s, makes the reported perlexity slightly overestimated
        unk_mask = tf.not_equal(targets, self.vocab.unk_index, name='unk_mask')
        mask = tf.cast(tf.logical_and(present_mask, unk_mask), tf.float32)
        output = tf.reshape(tf.concat(1, outputs), [-1, cfg.hidden_size])
        if self.training and cfg.softmax_samples < len(self.vocab.vocab):
            targets = tf.reshape(targets, [-1, 1])
            mask = tf.reshape(mask, [-1])
            loss = tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, output, targets,
                                              cfg.softmax_samples, len(self.vocab.vocab))
            loss *= mask
        else:
            logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(self.softmax_w),
                                              name='softmax_transform_mle'), self.softmax_b)
            loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                          [tf.reshape(targets, [-1])],
                                                          [tf.reshape(mask, [-1])])
        return tf.reshape(loss, [cfg.batch_size, -1])

    def discriminator_rnn(self, states):
        '''Recurrent discriminator that operates on the sequence of states of the sentences.'''
        with tf.variable_scope("Discriminator"):
            if cfg.d_rnn_bidirect:
                hidden_size = cfg.hidden_size
                fcell = self.rnn_cell(cfg.d_num_layers, hidden_size, return_states=True)
                bcell = self.rnn_cell(cfg.d_num_layers, hidden_size, return_states=True)
                seq_lengths = [cfg.max_sent_length] * (2 * cfg.batch_size)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(fcell, bcell, states,
                                                             sequence_length=seq_lengths,
                                                             swap_memory=True, dtype=tf.float32)
            else:
                hidden_size = cfg.hidden_size * 2
                cell = self.rnn_cell(cfg.d_num_layers, hidden_size, return_states=True)
                outputs, _ = tf.nn.dynamic_rnn(cell, states, swap_memory=True, dtype=tf.float32)
                outputs = (outputs,)  # to match bidirectional RNN's output format
            d_states = []
            for out in outputs:
                output = tf.slice(out, [0, 0, 0], [-1, -1, hidden_size])
                dir_states = tf.slice(out, [0, 0, hidden_size], [-1, -1, -1])
                # for GRU, we skipped the last layer states because they're the outputs
                d_states.append(tf.concat(2, [dir_states, output]))
        return self._discriminator_conv(tf.concat(2, d_states))

    def _discriminator_conv(self, states):
        '''Convolve output of bidirectional RNN and predict the discriminator label.'''
        with tf.variable_scope("Discriminator"):
            W_conv = tf.get_variable('W_conv', [cfg.d_conv_window, 1, states.get_shape()[2],
                                                cfg.hidden_size // cfg.d_conv_window],
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b_conv = tf.get_variable('b_conv', [cfg.hidden_size // cfg.d_conv_window],
                                     initializer=tf.constant_initializer(0.0))
            states = tf.expand_dims(states, 2)
            conv = tf.nn.conv2d(states, W_conv, strides=[1, 1, 1, 1], padding='SAME')
            conv_out = tf.reshape(conv, [2 * cfg.batch_size, -1,
                                         cfg.hidden_size // cfg.d_conv_window])
            conv_out = tf.nn.elu(tf.nn.bias_add(conv_out, b_conv))
            reduced = tf.reduce_mean(conv_out, [1])
            output = utils.linear(reduced, 1, True, 0.0, scope='discriminator_output')
        return output

    def discriminator_finalstate(self, states):
        '''Discriminator that operates on the final states of the sentences.'''
        with tf.variable_scope("Discriminator"):
            lin1 = tf.nn.elu(utils.linear(states[:, -1, :], cfg.hidden_size, True, 0.0,
                                          scope='discriminator_lin1'))
            lin2 = tf.nn.elu(utils.linear(lin1, cfg.hidden_size // 2, True, 0.0,
                                          scope='discriminator_lin2'))
            output = utils.linear(lin2, 1, True, 0.0, scope='discriminator_output')
        return output

    def discriminator_energy(self, states):
        '''An energy-based discriminator that tries to reconstruct the input states.'''
        with tf.variable_scope("Discriminator"):
            _, state = tf.nn.dynamic_rnn(self.rnn_cell(cfg.d_num_layers, cfg.hidden_size), states,
                                         swap_memory=True, dtype=tf.float32,
                                         scope='discriminator_encoder')
            # XXX use BiRNN+convnet for the encoder
            # this latent needs a more capacity than to reproduce the hidden states
            latent_size = cfg.hidden_size // 5
            latent = tf.nn.elu(utils.linear(state, latent_size, True,
                                            scope='discriminator_latent_transform'))
            latent = utils.highway(latent, layer_size=2, f=tf.nn.elu)
            decoder_input = tf.concat(1, [tf.zeros([2 * cfg.batch_size, 1, cfg.hidden_size]),
                                          states])
            decoder_input = tf.concat(2, [decoder_input,
                                          tf.tile(tf.expand_dims(latent, 1),
                                                  [1, decoder_input.get_shape()[1].value, 1])])
            output, _ = tf.nn.dynamic_rnn(self.rnn_cell(cfg.d_num_layers, cfg.hidden_size),
                                          decoder_input, swap_memory=True, dtype=tf.float32,
                                          scope='discriminator_decoder')
            output = tf.reshape(output, [-1, cfg.hidden_size])
            reconstructed = utils.linear(output, cfg.hidden_size, True, 0.0,
                                         scope='discriminator_reconst')
            reconstructed = tf.reshape(reconstructed, [2 * cfg.batch_size, -1, cfg.hidden_size])
        return reconstructed

    def gan_energy_loss(self, states, targets):
        '''Return the GAN energy loss. Put no variables here.'''
        losses = tf.reduce_sum(tf.square(states - targets), [1, 2]) / cfg.max_sent_length
        d_losses = losses[:cfg.batch_size] + tf.nn.relu(cfg.d_eb_margin - losses[cfg.batch_size:])
        g_losses = losses[cfg.batch_size:]
        return d_losses, g_losses

    def gan_loss(self, d_out, targets):
        '''Return the discriminator loss according to the label (1 if MLE mode).
           Put no variables here.'''
        return tf.nn.sigmoid_cross_entropy_with_logits(d_out, targets)

    def _train(self, cost, scope, optimizer, global_step=None):
        '''Generic training helper'''
        # TODO compute gradients more precisely with different costs and take one optimizer step
        #      after one global clip
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads = tf.gradients(cost, tvars)
        if cfg.max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, cfg.max_grad_norm)
        return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    def train_mle(self, cost):
        '''Training op for MLE mode.'''
        return self._train(cost, '.*/(Embeddings|Generator|MLE_Softmax)', self.g_optimizer,
                           self.global_step)

    def train_d(self, cost):
        '''Training op for GAN mode, discriminator.'''
        return self._train(cost, '.*/Discriminator', self.d_optimizer)

    def train_g(self, cost):
        '''Training op for GAN mode, generator.'''
        # don't update embeddings, just update the generated distributions
        return self._train(cost, '.*/Generator', self.g_optimizer)
