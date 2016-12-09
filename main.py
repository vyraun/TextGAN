import sys
import time

import numpy as np
import tensorflow as tf

from config import cfg
from reader import Reader, Vocab
from ngram import NGramModel
import utils


def call_session(session, model, batch):
    '''Use the session to run the model on the batch data.'''
    f_dict = {model.data: batch}
    # mle_train_op is tf.no_op() for a non-training model
    ops = [model.nll, model.mle_train_op]
    return session.run(ops, f_dict)[0]


def generate_sentences(session, model, vocab):
    '''Generate sentences using the generator.'''
#    f_dict = {model.data: np.zeros([cfg.batch_size, cfg.max_sent_length], dtype=np.int32)}
#    utils.display_sentences(session.run(model.generated, f_dict), vocab, cfg.char_model)
    pass  # TODO


def save_model(session, saver, perp, cur_iters):
    '''Save model file.'''
    save_file = cfg.save_file
    if not cfg.save_overwrite:
        save_file = save_file + '.' + str(cur_iters)
    print("Saving model (epoch perplexity: %.3f) ..." % perp)
    save_file = saver.save(session, save_file)
    print("Saved to", save_file)


def run_epoch(epoch, session, model, batch_loader, vocab, saver, steps, max_steps, gen_every):
    '''Runs the model on the given data for an epoch.'''
    start_time = time.time()
    nlls = 0.0
    shortterm_nlls = 0.0
    shortterm_steps = 0

    for step, batch in enumerate(batch_loader):
        cur_iters = steps + step
        nll = call_session(session, model, batch)
        nlls += nll
        shortterm_nlls += nll
        shortterm_steps += 1

        if step % cfg.print_every == 0:
            avg_nll = shortterm_nlls / shortterm_steps
            print("%d: %d (%d)  perplexity: %.3f  mle_loss: %.4f  speed: %.0f wps" %
                  (epoch + 1, step, cur_iters, np.exp(avg_nll), avg_nll,
                   shortterm_steps * cfg.batch_size / (time.time() - start_time)))

            shortterm_nlls = 0.0
            shortterm_steps = 0
            start_time = time.time()

        if gen_every > 0 and (step + 1) % gen_every == 0:
            for _ in range(cfg.gen_samples):
                generate_sentences(session, model, vocab)

        if saver is not None and cur_iters and cfg.save_every > 0 and \
                cur_iters % cfg.save_every == 0:
            save_model(session, saver, np.exp(nlls / step), cur_iters)

        if max_steps > 0 and cur_iters >= max_steps:
            break

    if gen_every < 0:
        for _ in range(cfg.gen_samples):
            generate_sentences(session, model, vocab)

    perp = np.exp(nlls / step)
    cur_iters = steps + step
    if saver is not None and cfg.save_every < 0:
        save_model(session, saver, perp, cur_iters)
    return perp, cur_iters


def main(_):
    vocab = Vocab()
    vocab.load_from_pickle()
    reader = Reader(vocab)

    config_proto = tf.ConfigProto()
    if not cfg.preallocate_gpu:
        config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
        with tf.variable_scope("Model") as scope:
            if cfg.training:
                with tf.variable_scope("LR"):
                    lr = tf.get_variable("lr", shape=[], initializer=tf.zeros_initializer,
                                         trainable=False)
                optimizer = utils.get_optimizer(lr, cfg.optimizer)
                model = NGramModel(vocab, True, optimizer=optimizer)
                scope.reuse_variables()
                eval_model = NGramModel(vocab, False)
            else:
                test_model = NGramModel(vocab, False)
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
            session.run(tf.assign(lr, cfg.learning_rate))
            for i in range(cfg.max_epoch):
                print("\nEpoch: %d" % (i + 1))
                perplexity, steps = run_epoch(i, session, model, reader.training(), vocab, saver,
                                              steps, cfg.max_steps, cfg.gen_every)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, perplexity))
                train_perps.append(perplexity)
                if cfg.validate_every > 0 and (i + 1) % cfg.validate_every == 0:
                    perplexity, _ = run_epoch(i, session, eval_model, reader.validation(), vocab,
                                              None, 0, -1, -1)
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
                                      cfg.max_steps, -1)
            print("Test Perplexity: %.3f" % perplexity)


if __name__ == "__main__":
    tf.app.run()
