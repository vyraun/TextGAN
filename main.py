import sys
import time

import numpy as np
import tensorflow as tf

from config import cfg
from reader import Reader, Vocab
from rnnlm import RNNLMModel
import utils


def call_mle_session(session, model, batch, use_gan):
    '''Use the session to run the model on the batch data.'''
    f_dict = {model.data: batch[0],
              model.data_dropped: batch[1],
              model.lengths: batch[2]}
    ops = [model.nll, model.mle_cost]
    # training ops are tf.no_op() for a non-training model
    train_ops = [model.mle_train_op]
    if use_gan:
        ops.append(model.d_cost)
        train_ops.append(model.d_train_op)
    ops.extend(train_ops)
    return session.run(ops, f_dict)[:-len(train_ops)]


def call_gan_session(session, model, generator=False):
    '''Use the session to train the generator of the GAN with fake samples.'''
    ops = [model.d_cost, model.g_cost]
    # train_ops will be tf.no_op() for a non-training model
    if generator:
        ops.append(model.g_train_op)
    else:
        ops.append(model.d_train_op)
    return session.run(ops)[:-1]


def generate_sentences(session, model, vocab):
    '''Generate sentences using the generator.'''
    # TODO remove this if nothing else is to be added here
    utils.display_sentences(session.run(model.generated), vocab, cfg.char_model)


def save_model(session, saver, perp, cur_iters):
    '''Save model file.'''
    save_file = cfg.save_file
    if not cfg.save_overwrite:
        save_file = save_file + '.' + str(cur_iters)
    print("Saving model (epoch perplexity: %.3f) ..." % perp)
    save_file = saver.save(session, save_file)
    print("Saved to", save_file)


def run_epoch(epoch, session, mle_model, gan_model, batch_loader, vocab,
              saver, steps, max_steps, scheduler, gen_every=0, lr_tracker=None):
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
    nod_steps = 0
    nog_steps = 0
    g_steps = 0
    d_steps = 0
    if cfg.d_energy_based:
        update_d = True
        update_g = True
    else:
        update_d = False
        update_g = False

    for step, batch in enumerate(batch_loader):
        if scheduler is not None:
            update_d = scheduler.update_d()
            update_g = scheduler.update_g()

        if lr_tracker is not None:
            lr_tracker.mle_mode()
        ret = call_mle_session(session, mle_model, batch, use_gan=update_d)
        nll, mle_cost = ret[:2]
        if update_d:
            d_cost = ret[-1]
        else:
            d_cost = None
        if update_d:
            d_steps += 1
            d1_cost, g1_cost = call_gan_session(session, gan_model)
        else:
            d1_cost = g1_cost = None
        if update_g:
            g_steps += 1
            if lr_tracker is not None:
                lr_tracker.gan_mode()
            d2_cost, g2_cost = call_gan_session(session, gan_model, generator=True)
        else:
            d2_cost = g2_cost = None
        d_costs = [c for c in [d_cost, d1_cost, d2_cost] if c is not None]
        g_costs = [c for c in [g1_cost, g2_cost] if c is not None]
        if not d_costs:
            nod_steps += 1
            d_cost = 0.0
        else:
            d_cost = np.mean(d_costs)
            if scheduler is not None:
                if cfg.d_energy_based:
                    d_acc = -1.0
                else:
                    d_acc = np.exp(-d_cost)
                scheduler.add_d_acc(d_acc)
        if not g_costs:
            nog_steps += 1
            g_cost = 0.0
        else:
            g_cost = np.mean(g_costs)

        if cfg.char_model:
            n_words = max(int((np.sum(batch[0] == vocab.vocab_lookup[' ']) / cfg.batch_size) + 1),
                          1)
        else:
            n_words = int(np.sum(batch[0] != 0) / cfg.batch_size)
        if scheduler is not None:
            scheduler.add_perp(np.exp(nll / n_words))
            scheduler.add_kld_weight(kld_weight)

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
            if shortterm_steps > nod_steps:
                avg_d_cost = shortterm_d_costs / (shortterm_steps - nod_steps)
                if cfg.d_energy_based:
                    d_acc = -1.0
                else:
                    d_acc = np.exp(-avg_d_cost)
            else:
                avg_d_cost = -1.0
                d_acc = -1.0
            if shortterm_steps > nog_steps:
                avg_g_cost = shortterm_g_costs / (shortterm_steps - nog_steps)
            else:
                avg_g_cost = -1.0
            print("%d: %d  perplexity: %.3f  mle_loss: %.4f  mle_cost: %.4f  d_cost: %.4f  "
                  "g_cost: %.4f  d_acc: %.4f  speed: %.0f wps  D:%d G:%d" % (epoch + 1, step,
                  np.exp(avg_nll), avg_nll, avg_mle_cost, avg_d_cost, avg_g_cost, d_acc,
                  shortterm_iters * cfg.batch_size / (time.time() - start_time), d_steps, g_steps))

            shortterm_nlls = 0.0
            shortterm_mle_costs = 0.0
            shortterm_d_costs = 0.0
            shortterm_g_costs = 0.0
            shortterm_iters = 0
            shortterm_steps = 0
            nod_steps = 0
            nog_steps = 0
            g_steps = 0
            d_steps = 0
            start_time = time.time()

        if gen_every > 0 and (step + 1) % gen_every == 0:
            for _ in range(cfg.gen_samples):
                generate_sentences(session, vocab)

        cur_iters = steps + step
        if saver is not None and cur_iters and cfg.save_every > 0 and \
                cur_iters % cfg.save_every == 0:
            save_model(session, saver, np.exp(nlls / iters), cur_iters)

        if max_steps > 0 and cur_iters >= max_steps:
            break

    if gen_every < 0:
        for _ in range(cfg.gen_samples):
            generate_sentences(session, gan_model, vocab)

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
        with tf.variable_scope("Model"):
            if cfg.training:
                with tf.variable_scope("LR"):
                    g_lr = tf.get_variable("g_lr", shape=[], initializer=tf.zeros_initializer,
                                           trainable=False)
                    d_lr = tf.get_variable("d_lr", shape=[], initializer=tf.zeros_initializer,
                                           trainable=False)
                lr_tracker = utils.LearningRateTracker(session, g_lr, d_lr)
                g_optimizer = utils.get_optimizer(g_lr, cfg.g_optimizer)
                d_optimizer = utils.get_optimizer(d_lr, cfg.d_optimizer)
                mle_model = RNNLMModel(vocab, True, True, None, None,
                                                g_optimizer=g_optimizer, d_optimizer=d_optimizer)
                gan_model = RNNLMModel(vocab, True, False, True, None,
                                                g_optimizer=g_optimizer, d_optimizer=d_optimizer)
                eval_mle_model = RNNLMModel(vocab, False, True, True, None)
                eval_gan_model = RNNLMModel(vocab, False, False, True, True)
            else:
                test_mle_model = RNNLMModel(vocab, False, True, None, None)
                test_gan_model = RNNLMModel(vocab, False, False, True, None)
        saver = tf.train.Saver()
        try:
            # try to restore a saved model file
            saver.restore(session, cfg.load_file)
            print("Model restored from", cfg.load_file)
        except ValueError:
            if cfg.training:
                tf.initialize_all_variables().run()
                print("No loadable model file, new model initialized.")
            else:
                print("You need to provide a valid model file for testing!")
                sys.exit(1)

        utils.list_all_variables()  # TODO remove
        print(1 // 0)

        if cfg.training:
            steps = 0
            train_perps = []
            valid_perps = []
            lr_tracker.update_mle_lr(cfg.mle_learning_rate)
            lr_tracker.update_g_lr(cfg.g_learning_rate)
            lr_tracker.update_d_lr(cfg.d_learning_rate)
            scheduler = utils.Scheduler(cfg.min_d_acc, cfg.max_d_acc, cfg.max_perplexity,
                                        cfg.sc_list_size, cfg.sc_decay)
            for i in range(cfg.max_epoch):
                print("\nEpoch: %d" % (i + 1))
                perplexity, steps = run_epoch(i, session, mle_model, gan_model,
                                              reader.training(), vocab, saver, steps,
                                              cfg.max_steps, scheduler, gen_every=cfg.gen_every,
                                              lr_tracker=lr_tracker)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, perplexity))
                train_perps.append(perplexity)
                if cfg.validate_every > 0 and (i + 1) % cfg.validate_every == 0:
                    perplexity, _ = run_epoch(i, session, eval_mle_model, eval_gan_model,
                                              reader.validation(), vocab,
                                              None, 0, -1, None, gen_every=-1)
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
            perplexity, _ = run_epoch(0, session, test_mle_model, test_gan_model,
                                      reader.testing(), vocab, None, 0, cfg.max_steps, None,
                                      gen_every=-1)
            print("Test Perplexity: %.3f" % perplexity)


if __name__ == "__main__":
    tf.app.run()
