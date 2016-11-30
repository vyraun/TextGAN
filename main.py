import sys
import time

import numpy as np
import tensorflow as tf

from config import cfg
from encdec import EncoderDecoderModel
from reader import Reader, Vocab
import utils


def call_session(session, model, batch, train_d=False, train_g=False, get_latent=False):
    '''Use the session to run the model on the batch data.'''
    f_dict = {model.data: batch[0],
              model.data_dropped: batch[1],
              model.lengths: batch[2]}
    ops = [model.nll, model.kld, model.mle_cost, model.kld_weight, model.d_cost, model.g_cost]
    # training ops are tf.no_op() for a non-training model
    train_ops = [model.mle_train_op]
    if train_d:
        train_ops.append(model.d_train_op)
    if train_g:
        train_ops.append(model.g_train_op)
    if get_latent:
        ops.append(model.latent)
    ops.extend(train_ops)
    return session.run(ops, f_dict)[:-len(train_ops)]


def generate_sentences(session, model, vocab, random_dims=None, mle_generator=False,
                       true_output=None):
    '''Generate sentences using the generator, either novel or from known encodings (mle_generator).
    '''
    # TODO for MAP inference, use beam search
    # XXX uncond:
    #f_dict = {model.data: np.zeros([cfg.batch_size, cfg.max_sent_length], dtype=np.int32)}
    #utils.display_sentences(session.run(model.generated, f_dict), vocab, cfg.char_model)
    if mle_generator:
        print('\nTrue output')
        utils.display_sentences(true_output[:, 1:], vocab, cfg.char_model)
        print('Sentences generated from true encodings')
        f_dict = {model.latent: random_dims}
    else:
        print('\nNovel sentences: new batch')
        f_dict = {}
    output = session.run(model.generated, f_dict)
    utils.display_sentences(output, vocab, cfg.char_model)


def save_model(session, saver, perp, cur_iters):
    '''Save model file.'''
    save_file = cfg.save_file
    if not cfg.save_overwrite:
        save_file = save_file + '.' + str(cur_iters)
    print("Saving model (epoch perplexity: %.3f) ..." % perp)
    save_file = saver.save(session, save_file)
    print("Saved to", save_file)


def run_epoch(epoch, session, model, generator, batch_loader, vocab,
              saver, steps, max_steps, scheduler, use_gan, gen_every):
    '''Runs the model on the given data for an epoch.'''
    start_time = time.time()
    nlls = 0.0
    klds = 0.0
    mle_costs = 0.0
    iters = 0
    shortterm_nlls = 0.0
    shortterm_klds = 0.0
    shortterm_mle_costs = 0.0
    shortterm_d_costs = 0.0
    shortterm_g_costs = 0.0
    shortterm_iters = 0
    shortterm_steps = 0
    g_steps = 0
    d_steps = 0
    gan_steps = 0
    latest_latent = None
    update_d = False
    update_g = False
    update_gan = False

    for step, batch in enumerate(batch_loader):
        if scheduler is not None:
            update_d = use_gan and scheduler.update_d()
            update_g = use_gan and scheduler.update_g()
            update_gan = update_d or update_g
            if update_d:
                d_steps += 1
            if update_g:
                g_steps += 1
            if update_gan:
                gan_steps += 1
        if gen_every > 0 and (step + 1) % gen_every == 0:
            get_latent = True
        else:
            get_latent = False

        ret = call_session(session, model, batch, train_d=update_d, train_g=update_g,
                           get_latent=get_latent)
        nll, kld, mle_cost, kld_weight, d_cost, g_cost = ret[:6]
        ret = ret[6:]
        if get_latent:
            latest_latent = ret.pop(0)
        if not update_gan:
            d_cost = 0.0
            g_cost = 0.0
        elif scheduler is not None:
            if cfg.d_energy_based:
                d_acc = -1.0
            else:
                d_acc = np.exp(-d_cost)
            scheduler.add_d_acc(d_acc)

        if cfg.char_model:
            n_words = (np.sum(batch[0] == vocab.vocab_lookup[' ']) // cfg.batch_size) + 1
        else:
            n_words = max(np.sum(batch[0] != 0) // cfg.batch_size, 1)
        if scheduler is not None:
            scheduler.add_perp(np.exp(nll / n_words))
            scheduler.add_kld_weight(kld_weight)

        nlls += nll
        klds += kld
        mle_costs += mle_cost
        shortterm_nlls += nll
        shortterm_klds += kld
        shortterm_mle_costs += mle_cost
        shortterm_d_costs += d_cost
        shortterm_g_costs += g_cost
        iters += n_words
        shortterm_iters += n_words
        shortterm_steps += 1

        if step % cfg.print_every == 0:
            avg_nll = shortterm_nlls / shortterm_iters
            avg_kld = shortterm_klds / shortterm_steps
            avg_mle_cost = shortterm_mle_costs / shortterm_steps
            if gan_steps:
                avg_d_cost = shortterm_d_costs / gan_steps
                if cfg.d_energy_based:
                    d_acc = -1.0
                else:
                    d_acc = np.exp(-avg_d_cost)
            else:
                avg_d_cost = -1.0
                d_acc = -1.0
            if g_steps:
                avg_g_cost = shortterm_g_costs / g_steps
            else:
                avg_g_cost = -1.0
            print("%d: %d  perplexity: %.3f  mle_loss: %.4f  kld_loss: %.4f  mle_cost: %.4f  "
                  "kld_weight: %.4f  d_cost: %.4f  g_cost: %.4f  d_acc: %.4f  speed: %.0f wps  "
                  "D:%d G:%d" %
                  (epoch + 1, step, np.exp(avg_nll), avg_nll, avg_kld, avg_mle_cost, kld_weight,
                   avg_d_cost, avg_g_cost, d_acc,
                   shortterm_iters * cfg.batch_size / (time.time() - start_time), d_steps, g_steps))

            shortterm_nlls = 0.0
            shortterm_klds = 0.0
            shortterm_mle_costs = 0.0
            shortterm_d_costs = 0.0
            shortterm_g_costs = 0.0
            shortterm_iters = 0
            shortterm_steps = 0
            g_steps = 0
            d_steps = 0
            gan_steps = 0
            start_time = time.time()

        if gen_every > 0 and (step + 1) % gen_every == 0:
            if latest_latent is not None:
                generate_sentences(session, generator, vocab, latest_latent, True, batch[0])
            for _ in range(cfg.gen_samples):
                generate_sentences(session, model, vocab)

        cur_iters = steps + step
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
                model = EncoderDecoderModel(vocab, True, use_gan=cfg.use_gan,
                                            g_optimizer=g_optimizer, d_optimizer=d_optimizer)
                scope.reuse_variables()
                eval_model = EncoderDecoderModel(vocab, False, use_gan=cfg.use_gan)
            else:
                test_model = EncoderDecoderModel(vocab, False, use_gan=cfg.use_gan)
                scope.reuse_variables()
            generator = EncoderDecoderModel(vocab, False, use_gan=cfg.use_gan, generator=True)
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

        if cfg.training:
            steps = 0
            train_perps = []
            valid_perps = []
            session.run(tf.assign(g_lr, cfg.g_learning_rate))
            session.run(tf.assign(d_lr, cfg.d_learning_rate))
            if cfg.sc_use_kld_weight:
                min_kld_weight = cfg.anneal_max - 1e-4
            else:
                min_kld_weight = -1
            scheduler = utils.Scheduler(cfg.min_d_acc, cfg.max_d_acc, cfg.max_perplexity,
                                        min_kld_weight, cfg.sc_list_size, cfg.sc_decay)
            for i in range(cfg.max_epoch):
                print("\nEpoch: %d" % (i + 1))
                perplexity, steps = run_epoch(i, session, model, generator,
                                              reader.training(), vocab, saver, steps,
                                              cfg.max_steps, scheduler, cfg.use_gan, cfg.gen_every)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, perplexity))
                train_perps.append(perplexity)
                if cfg.validate_every > 0 and (i + 1) % cfg.validate_every == 0:
                    perplexity, _ = run_epoch(i, session, eval_model, generator,
                                              reader.validation(), vocab,
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
            perplexity, _ = run_epoch(0, session, test_model, generator,
                                      reader.testing(), vocab, None, 0, cfg.max_steps, None,
                                      cfg.use_gan, -1)
            print("Test Perplexity: %.3f" % perplexity)


if __name__ == "__main__":
    tf.app.run()
