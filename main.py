from __future__ import division

import numpy as np
import tensorflow as tf

from config import Config
import reader


class GANModel(object):
    """The GAN model."""

    def __init__(self, config):
        batch_size = config.batch_size
        num_steps = config.num_steps


    def rnn_cell(self, config):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size)
        if config.training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob,
                                                      name='lstm_dropout')
        return tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)


    def word_embeddings(self, config, vocab):
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("word_embedding", [config.vocab_size,
                                                           config.word_emb_size],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))
            self.word_embedding = embedding
            if config.pretrained_emb:
                cembedding = tf.constant(vocab.embeddings, dtype=embedding.dtype,
                                         name="pre_word_embedding")
                embedding = tf.concat(1, [embedding, cembedding], name='concat_word_embeddings')
            inputs = tf.nn.embedding_lookup(embedding, self.input_data,
                                            name='word_embedding_lookup')
        return inputs


    def rnn(self, inputs, structured_inputs, cell, config):
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            if config.conditional:
                emb_size = sum(config.mimic_embeddings.values())
                assert emb_size >= config.hidden_size
                transform_w = tf.get_variable("struct_transform_w", [emb_size, config.hidden_size],
                                              initializer=tf.contrib.layers.xavier_initializer())
                transform_b = tf.get_variable("struct_transform_b", [config.hidden_size],
                                              initializer=tf.zeros_initializer)
                structured_inputs = tf.nn.bias_add(tf.matmul(structured_inputs, transform_w,
                                                             name='transform_structs'),
                                                   transform_b)

            for time_step in xrange(config.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                if config.conditional:
                    # TODO: update gate design (see ff)
                    # state is:              batch_size x 2 * size * num_layers
                    # structured_inputs is:  batch_size x size
                    gate_w = tf.get_variable("struct_gate_w",
                                             [2 * config.hidden_size * config.num_layers,
                                              config.hidden_size],
                                             initializer=tf.contrib.layers.xavier_initializer())
                    gate_b = tf.get_variable("struct_gate_b", [config.hidden_size],
                                             initializer=tf.ones_initializer)
                    gate = tf.sigmoid(tf.nn.bias_add(tf.matmul(state, gate_w,
                                                               name='gate_transform'), gate_b))
                    outputs.append((gate * cell_output) + ((1.0 - gate) * structured_inputs))
                else:
                    outputs.append(cell_output)
        return outputs, state


    def rnn_loss(self, outputs, config):
        output = tf.reshape(tf.concat(1, outputs), [-1, config.hidden_size])
        softmax_w = tf.get_variable("softmax_w", [config.vocab_size, config.hidden_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
        softmax_b = tf.get_variable("softmax_b", [config.vocab_size],
                                    initializer=tf.zeros_initializer)
        if config.training and config.softmax_samples < config.vocab_size:
            targets = tf.reshape(self.targets, [-1, 1])
            mask = tf.reshape(self.mask, [-1])
            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, output, targets,
                                              config.softmax_samples, config.vocab_size) * mask
        else:
            logits = tf.nn.bias_add(tf.matmul(output, tf.transpose(softmax_w),
                                              name='softmax_transform'), softmax_b)
            loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                          [tf.reshape(self.targets, [-1])],
                                                          [tf.reshape(self.mask, [-1])])
        return tf.reshape(loss, [config.batch_size, config.num_steps])


    def prepare(self, config, vocab):
        if config.recurrent:
            cell = self.rnn_cell(config)
            self.initial_state = cell.zero_state(config.batch_size, tf.float32)

        if config.recurrent:
            outputs, self.final_state = self.rnn(inputs, structured_inputs, cell, config)
            self.loss = self.rnn_loss(outputs, config)

        self.perplexity = tf.reduce_sum(self.loss) / config.batch_size
        self.additional = tf.zeros([])
        if config.conditional:
            self.additional += config.struct_l1_weight * self.struct_l1
            self.additional += config.struct_l2_weight * self.struct_l2
        self.cost = self.perplexity + self.additional
        if config.training:
            self.train_op = self.train(config)
        else:
            self.train_op = tf.no_op()


    def assign_lr(self, session, lr_value, verbose=True):
        if verbose:
            print "Setting learning rate to", lr_value
        session.run(tf.assign(self.lr, lr_value))


    def train(self, config):
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
        grads = tf.gradients(self.cost, tvars)
        if config.recurrent and config.max_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
        return optimizer.apply_gradients(zip(grads, tvars))


def call_session(session, m, config, vocab, prev_state, zero_state, batch, profile_kwargs):
    x, y, mask, aux, aux_len, new_batch = batch
    f_dict = {m.input_data: x, m.targets: y}
    if config.recurrent:
        f_dict[m.mask] = mask
        if new_batch:
            f_dict[m.initial_state] = zero_state
        else:
            f_dict[m.initial_state] = prev_state
    if config.conditional:
        for feat, vals in aux.items():
            f_dict[m.aux_data[feat]] = vals
            f_dict[m.aux_data_len[feat]] = aux_len[feat]
        if config.inspect == 'struct':
            for v in m.struct_enable.values():
                f_dict[v] = 1.0
    if config.recurrent:
        ops = [m.perplexity, m.struct_l1, m.struct_l2, m.cost, m.final_state, m.train_op]
    else:
        ops = [m.perplexity, m.struct_l1, m.struct_l2, m.cost, m.train_op]
        if config.distance_dep:
            ops += m.transforms
        if config.dump_results_file or (config.conditional and config.inspect == 'struct'):
            ops += [m.loss, m.gate]

    ret = session.run(ops, f_dict, **profile_kwargs)

    if not config.recurrent:
        if config.dump_results_file or (config.conditional and config.inspect == 'struct'):
            loss, gate = ret[-2:]
            ret = ret[:-2]

            losses = [[] for _ in xrange(config.batch_size)]
            for i in xrange(config.batch_size):
                if config.conditional:
                    losses[i].append((loss[i], 'all', gate[i]))
                else:
                    losses[i].append((loss[i], 'unconditional', gate[i]))

        #inspect before returning
        if config.conditional and config.inspect == 'struct':
            feats = m.struct_enable.keys()
            for v in m.struct_enable.values():
                f_dict[v] = 0.0
            loss, gate = session.run([m.loss, m.gate], f_dict)
            for i in xrange(config.batch_size):
                losses[i].append((loss[i], 'none', gate[i]))
            for only in [True, False]:
                for feat in feats:
                    for v in m.struct_enable.values():
                        if only:
                            f_dict[v] = 0.0
                        else:
                            f_dict[v] = 1.0
                    if only:
                        f_dict[m.struct_enable[feat]] = 1.0
                    else:
                        f_dict[m.struct_enable[feat]] = 0.0
                    if only:
                        s = 'only_'
                    else:
                        s = 'no_'
                    loss, gate = session.run([m.loss, m.gate], f_dict)
                    for i in xrange(config.batch_size):
                        losses[i].append((loss[i], s+feat, gate[i]))

        if config.dump_results_file:
            global write_results
            if config.conditional:
                write_results.append((x, y, losses, aux, aux_len))
            else:
                write_results.append((x, y, losses))

        transforms = []
        if config.distance_dep:
            transforms = ret[-config.num_steps:]
            ret = ret[:-config.num_steps]

        return ret[:-1] + [transforms]

    else: # recurrent
        return ret[:-1]


def run_epoch(session, m, config, vocab, saver, steps, run_options, run_metadata, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    perps = 0.0
    costs = 0.0
    iters = 0
    shortterm_perps = 0.0
    shortterm_l1s = 0.0
    shortterm_l2s = 0.0
    shortterm_costs = 0.0
    shortterm_iters = 0
    batches = 0
    if config.recurrent:
        zero_state = m.initial_state.eval()
        state = None
    for step, batch in enumerate(reader.mimic_iterator(config, vocab)):
        profile_kwargs = {}
        if config.profile:
            profile_kwargs['options'] = run_options
            profile_kwargs['run_metadata'] = run_metadata

        if config.recurrent:
            perp, l1, l2, cost, state = call_session(session, m, config, vocab, state, zero_state,
                                                     batch, profile_kwargs)
        else:
            perp, l1, l2, cost, transforms = call_session(session, m, config, vocab, None, None,
                                                          batch, profile_kwargs)

        if config.profile:
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(config.timeline_file, 'w') as f:
                f.write(ctf)
            config.profile = False

        perps += perp
        costs += cost
        shortterm_perps += perp
        shortterm_l1s += l1
        shortterm_l2s += l2
        shortterm_costs += cost
        if config.recurrent:
            iters += config.num_steps
            shortterm_iters += config.num_steps
        else:
            iters += 1
            shortterm_iters += 1

        if verbose and step % config.print_every == 0:
            avg_perp = shortterm_perps / shortterm_iters
            avg_l1 = shortterm_l1s / shortterm_iters
            avg_l2 = shortterm_l2s / shortterm_iters
            avg_cost = shortterm_costs / shortterm_iters
            if config.recurrent:
                print("%d  perplexity: %.3f  ml_loss: %.4f  struct_l1: %.4f  struct_l2: %.4f  " \
                      "cost: %.4f  speed: %.0f wps" %
                      (step, np.exp(avg_perp), avg_perp, avg_l1, avg_l2, avg_cost,
                       shortterm_iters * config.batch_size / (time.time() - start_time)))
            else:
                print("%d  perplexity: %.3f  ml_loss: %.4f  struct_l1: %.4f  struct_l2: %.4f  " \
                      "cost: %.4f  speed: %.0f wps %.0f pps" %
                      (step, np.exp(avg_perp), avg_perp, avg_l1, avg_l2, avg_cost,
                       shortterm_iters * config.num_steps * config.batch_size / (time.time() - \
                                                                                 start_time),
                       shortterm_iters * config.batch_size / (time.time() - start_time)))
                if config.distance_dep:
                    print "        position transforms ",
                    for pos in xrange(config.num_steps):
                        print ' norm%d: %.3f' % (pos, np.linalg.norm(transforms[pos])),
                    print
            shortterm_perps = 0.0
            shortterm_l1s = 0.0
            shortterm_l2s = 0.0
            shortterm_costs = 0.0
            shortterm_iters = 0
            start_time = time.time()

        cur_iters = steps + step
        if config.training and step and step % config.save_every == 0:
            save_file = config.save_file
            if not config.save_overwrite:
                save_file = save_file + '.' + str(cur_iters)
            if verbose: print "Saving model (epoch perplexity: %.3f) ..." % np.exp(perps / iters)
            save_file = saver.save(session, save_file)
            if verbose: print "Saved to", save_file

        if cur_iters >= config.max_steps:
            break
        elif config.training and cur_iters >= config.decay_step and not m.decayed:
            m.assign_lr(session, config.learning_rate2)
            m.decayed = True

    return np.exp(perps / iters), steps + step


def main(_):
    config = Config()
    vocab = reader.Vocab(config)

    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        with tf.variable_scope("model", reuse=None):
            m = LMModel(config=config)
            m.prepare(config, vocab)
        saver = tf.train.Saver()
        try:
            saver.restore(session, config.load_file)
            print "Model restored from", config.load_file
        except ValueError:
            if config.training:
                tf.initialize_all_variables().run()
                print "No loadable model file, new model initialized."
            else:
                print "You need to provide a valid model file for testing!"
                sys.exit(1)

        if not config.struct_only:
            emb_saver = tf.train.Saver([m.word_embedding])
            try:
                emb_saver.restore(session, config.load_emb_file)
                print "Word embeddings restored from", config.load_emb_file
            except ValueError:
                pass
        if config.conditional:
            struct_saver = tf.train.Saver(m.struct_embeddings)
            try:
                struct_saver.restore(session, config.load_struct_file)
                print "Structured embeddings restored from", config.load_struct_file
            except ValueError:
                pass

        steps = 0
        if config.training:
            m.assign_lr(session, config.learning_rate)
        for i in xrange(config.max_epoch):
            if i+1 == config.decay_epoch and not m.decayed:
                m.assign_lr(session, config.learning_rate2)
                m.decayed = True
            if config.training:
                print "Epoch: %d Learning rate: %.4f" % (i + 1, session.run(m.lr))
            perplexity, steps = run_epoch(session, m, config, vocab, saver, steps, run_options,
                                          run_metadata, verbose=True)
            if config.training:
                print "Epoch: %d Train Perplexity: %.3f" % (i + 1, perplexity)
            else:
                print "Test Perplexity: %.3f" % (perplexity,)
                break
            if steps >= config.max_steps:
                break


if __name__ == "__main__":
    tf.app.run()
