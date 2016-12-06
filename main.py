import sys
import time

import numpy as np
import tensorflow as tf

from config import cfg
from reader import Reader, Vocab
from rnnlm import RNNLMModel
import utils


def call_session(session, model, batch, train_d=False, train_g=False):
    '''Use the session to run the model on the batch data.'''
    f_dict = {model.data: batch}
    ops = [model.nll, model.mle_cost, model.d_cost, model.g_cost]
    # training ops are tf.no_op() for a non-training model
    train_ops = [model.mle_train_op]
    if train_d:
        train_ops.append(model.d_train_op)
    if train_g:
        train_ops.append(model.g_train_op)
    ops.extend(train_ops)
    return session.run(ops, f_dict)[:-len(train_ops)]


def generate_sentences(session, model, vocab):
    '''Generate sentences using the generator.'''
    f_dict = {model.data: np.zeros([cfg.batch_size, cfg.max_sent_length], dtype=np.int32)}
    utils.display_sentences(session.run(model.generated, f_dict), vocab, cfg.char_model)


def save_model(session, saver, perp, cur_iters):
    '''Save model file.'''
    save_file = cfg.save_file
    if not cfg.save_overwrite:
        save_file = save_file + '.' + str(cur_iters)
    print("Saving model (epoch perplexity: %.3f) ..." % perp)
    save_file = saver.save(session, save_file)
    print("Saved to", save_file)


def run_epoch(epoch, session, model, batch_loader, vocab, saver, steps, max_steps, scheduler,
              use_gan, gen_every):
    '''Runs the model on the given data for an epoch.'''
    start_time = time.time()
    nlls = 0.0
    mle_costs = 0.0
    iters = 0
    shortterm_nlls = 0.0
    shortterm_mle_costs = 0.0
    shortterm_d_costs = 0.0
    shortterm_g_costs = 0.0
    shortterm_iters = 0
    shortterm_steps = 0
    g_steps = 0
    d_steps = 0
    update_d = False
    update_g = False

    for step, batch in enumerate(batch_loader):
        cur_iters = steps + step
        if scheduler is not None:
            update_d = use_gan and scheduler.update_d()
            update_g = use_gan and scheduler.update_g()
        if update_d:
            d_steps += 1
        if update_g:
            g_steps += 1

        nll, mle_cost, d_cost, g_cost = call_session(session, model, batch, train_d=update_d,
                                                     train_g=update_g)
        if scheduler is not None:
            if cfg.d_energy_based:
                d_acc = -1.0
            else:
                d_acc = np.exp(-d_cost)
            scheduler.add_d_acc(d_acc)

        if cfg.char_model:
            n_words = (np.sum(batch == vocab.vocab_lookup[' ']) // cfg.batch_size) + 1
        else:
            n_words = cfg.max_sent_length
        if scheduler is not None:
            scheduler.add_perp(np.exp(nll / n_words))

        nlls += nll
        mle_costs += mle_cost
        shortterm_nlls += nll
        shortterm_mle_costs += mle_cost
        shortterm_d_costs += d_cost
        shortterm_g_costs += g_cost
        iters += n_words
        shortterm_iters += n_words
        shortterm_steps += 1

        if step % cfg.print_every == 0:
            avg_nll = shortterm_nlls / shortterm_iters
            avg_mle_cost = shortterm_mle_costs / shortterm_steps
            avg_d_cost = shortterm_d_costs / shortterm_steps
            if cfg.d_energy_based:
                d_acc = -1.0
            else:
                d_acc = np.exp(-avg_d_cost)
            if g_steps:
                avg_g_cost = shortterm_g_costs / g_steps
            else:
                avg_g_cost = -1.0
            print("%d: %d (%d)  perplexity: %.3f  mle_loss: %.4f  mle_cost: %.4f  d_cost: %.4f  "
                  "g_cost: %.4f  d_acc: %.4f  speed: %.0f wps  D:%d G:%d" % (epoch + 1, step,
                  cur_iters, np.exp(avg_nll), avg_nll, avg_mle_cost, avg_d_cost, avg_g_cost, d_acc,
                  shortterm_iters * cfg.batch_size / (time.time() - start_time), d_steps, g_steps))

            shortterm_nlls = 0.0
            shortterm_mle_costs = 0.0
            shortterm_d_costs = 0.0
            shortterm_g_costs = 0.0
            shortterm_iters = 0
            shortterm_steps = 0
            g_steps = 0
            d_steps = 0
            start_time = time.time()

        if gen_every > 0 and (step + 1) % gen_every == 0:
            for _ in range(cfg.gen_samples):
                generate_sentences(session, model, vocab)

        if saver is not None and cur_iters and cfg.save_every > 0 and \
                cur_iters % cfg.save_every == 0:
            save_model(session, saver, np.exp(nlls / iters), cur_iters)

        if max_steps > 0 and cur_iters >= max_steps:
            break

    if gen_every < 0:
        for _ in range(cfg.gen_samples):
            generate_sentences(session, model, vocab)

    perp = np.exp(nlls / iters)
    cur_iters = steps + step
    if saver is not None and cfg.save_every < 0:
        save_model(session, saver, perp, cur_iters)
    return perp, cur_iters


def main(_):
    vocab = Vocab()
    vocab.load_from_pickle()
    reader = Reader(vocab)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        with tf.variable_scope("Model") as scope:
            if cfg.training:
                with tf.variable_scope("LR"):
                    g_lr = tf.get_variable("g_lr", shape=[], initializer=tf.zeros_initializer,
                                           trainable=False)
                    d_lr = tf.get_variable("d_lr", shape=[], initializer=tf.zeros_initializer,
                                           trainable=False)
                g_optimizer = utils.get_optimizer(g_lr, cfg.g_optimizer)
                d_optimizer = utils.get_optimizer(d_lr, cfg.d_optimizer)
                model = RNNLMModel(vocab, True, cfg.use_gan, g_optimizer=g_optimizer,
                                   d_optimizer=d_optimizer)
                scope.reuse_variables()
                eval_model = RNNLMModel(vocab, False, cfg.use_gan)
            else:
                test_model = RNNLMModel(vocab, False, cfg.use_gan)
        saver = tf.train.Saver()
        steps = 0
        try:
            # try to restore a saved model file
            saver.restore(session, cfg.load_file)
            print("Model restored from", cfg.load_file)
            steps = session.run(model.global_step)
        except ValueError:
            if cfg.training:
                tf.initialize_all_variables().run()
                print("No loadable model file, new model initialized.")
            else:
                print("You need to provide a valid model file for testing!")
                sys.exit(1)

        if cfg.training:
            train_perps = []
            valid_perps = []
            session.run(tf.assign(g_lr, cfg.g_learning_rate))
            session.run(tf.assign(d_lr, cfg.d_learning_rate))
            scheduler = utils.Scheduler(cfg.min_d_acc, cfg.max_d_acc, cfg.max_perplexity,
                                        cfg.sc_list_size, cfg.sc_decay)
            for i in range(cfg.max_epoch):
                print("\nEpoch: %d" % (i + 1))
                perplexity, steps = run_epoch(i, session, model, reader.training(), vocab, saver,
                                              steps, cfg.max_steps, scheduler, cfg.use_gan,
                                              cfg.gen_every)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, perplexity))
                train_perps.append(perplexity)
                if cfg.validate_every > 0 and (i + 1) % cfg.validate_every == 0:
                    perplexity, _ = run_epoch(i, session, eval_model, reader.validation(), vocab,
                                              None, 0, -1, None, cfg.use_gan, -1)
                    print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, perplexity))
                    valid_perps.append(perplexity)
                else:
                    valid_perps.append(None)
                print('Train:', train_perps)
                print('Valid:', valid_perps)
                if steps >= cfg.max_steps:
                    break
        else:
            print('\nTesting')
            perplexity, _ = run_epoch(0, session, test_model, reader.testing(), vocab, None, 0,
                                      cfg.max_steps, None, cfg.use_gan, -1)
            print("Test Perplexity: %.3f" % perplexity)


if __name__ == "__main__":
    tf.app.run()
