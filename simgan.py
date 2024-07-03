import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers, layers, Model
from os import path, makedirs
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import H5DatasetIter, HistoryBuffer
from models import refiner_model, discriminator_model
from datetime import datetime

INPUT_SHAPE = (35, 55, 1)

# INFO: Hyper params
LEARNING_RATE = 0.001
# TODO: find out approp value for this
LAMBDA = 0.0001 # Self Regularization Loss lambda param
BATCH_SIZE = 512
# TODO: Approp size for HistoryBuffer ?
HIST_BUFFER_SIZE = 100 * BATCH_SIZE
STEPS = 10000
D_UPDATE_PER_STEP = 1
R_UPDATE_PER_STEP = 2
PRETRAIN_REFINER_STEPS = 1000
PRETRAIN_DISC_STEPS = 1000

# save model every 100 steps
DEBUG_INTERVAL = 100
N_PLOG_IMGS = 10

time_str = datetime.now().strftime("%d_%m_%y_%H%M")
cache_dir = path.join("cache", time_str)
if not path.isdir(cache_dir):
    makedirs(cache_dir)

tb_writer_path = path.join(cache_dir, "tb_logs")
tb_writer = tf.summary.create_file_writer(tb_writer_path)
print('cache_dir', cache_dir)
print('tb logs path', tb_writer_path)

def self_regularization_loss(y_t, y_p):
    return tf.multiply(LAMBDA, tf.reduce_sum(tf.abs(y_p - y_t)))

def local_adversarial_loss(y_t, y_p):
    y_t = tf.reshape(y_t, (-1, 2))
    y_p = tf.reshape(y_p, (-1, 2))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_t, logits=y_p))

def get_cl_args():
    parser = ArgumentParser(description="Training", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("real_ds_h5", type=str, help="Real dataset h5 file")
    parser.add_argument("syn_ds_h5", type=str, help="Synthetic dataset h5 file")
    parser.add_argument('--refiner_model', default=None, type=str, help="Refiner model to preload")
    parser.add_argument('--discriminator_model', default=None, type=str, help="Descriminator model to preload")
    return parser.parse_args()

def pretrain_refiner(R, syn_ds):
    print(f"Pre-training refiner for {PRETRAIN_REFINER_STEPS} steps")

    for i in range(1, PRETRAIN_REFINER_STEPS+1):
        print(f"Refiner pretrain step {i}", end=" ")
        syn_batch = next(syn_ds)
        b_loss = R.train_on_batch(syn_batch, syn_batch)
        tf.summary.scalar('refiner_pretrain_loss', b_loss, step=i)
        print(f"Loss = {b_loss}")

        if (i % DEBUG_INTERVAL) == 0:
            with tb_writer.as_default():
                tf.summary.image("Refiner pretrain",
                    (R.predict_on_batch(syn_batch[:N_PLOG_IMGS]) + 1.0) / 2.0,
                        max_outputs=N_PLOG_IMGS, step=i)

    save_path = path.join(cache_dir, "R_pretrained.keras")
    R.save(save_path)
    print(f"Refiner pretrained saved at {save_path}")

def pretrain_discriminator(D, R, real_ds, syn_ds):
    print(f"Pre-training discrminator for {PRETRAIN_DISC_STEPS} steps")

    y_real = np.array( [ [[1.0, 0.0]] * D.output_shape[1] ] * BATCH_SIZE )
    y_refined = np.array( [ [[0.0, 1.0]] * D.output_shape[1] ] * BATCH_SIZE )
    print(y_real.shape, y_refined.shape)

    for i in range(1, PRETRAIN_DISC_STEPS+1):
        print(f"Descriminator pretrain step {i}", end=" ")
        real_batch = next(real_ds)
        real_loss = D.train_on_batch(real_batch, y_real)
        syn_batch = next(syn_ds)
        refined_batch = R.predict_on_batch(syn_batch)
        ref_loss = D.train_on_batch(refined_batch, y_refined)
        with tb_writer.as_default():
            tf.summary.scalar('discriminator_pretrain_real_loss', real_loss, step=i)
            tf.summary.scalar('discriminator_pretrain_ref_loss', ref_loss, step=i)
        print(f"Real Loss = {real_loss} Refined Loss = {ref_loss}")

    save_path = path.join(cache_dir, "D_pretrained.keras")
    D.save(save_path)
    print(f"Disc pretrained with saved at {save_path}")

def adversarial_training(R, D, combined, syn_ds, real_ds):
    hist_buffer = HistoryBuffer(INPUT_SHAPE, HIST_BUFFER_SIZE, BATCH_SIZE)

    y_real = np.array( [ [[1.0, 0.0]] * D.output_shape[1] ] * BATCH_SIZE )
    y_refined = np.array( [ [[0.0, 1.0]] * D.output_shape[1] ] * BATCH_SIZE )
    print(y_real.shape, y_refined.shape)

    for i in range(1, STEPS+1):
        print(f"Step: {i}")

        for _ in range(R_UPDATE_PER_STEP):
            syn_batch = next(syn_ds)
            loss = combined.train_on_batch(syn_batch, [syn_batch, y_real])
            with tb_writer.as_default():
                tf.summary.scalar("self_reg_loss", loss[0], step=i)
                tf.summary.scalar("lcl_adv_loss", loss[1], step=i)
            print(f"self_reg_loss={loss[0]} lcl_adv_loss={loss[1]}")

        for _ in range(D_UPDATE_PER_STEP):
            syn_batch = next(syn_ds)
            real_batch = next(real_ds)

            refined_batch = R.predict_on_batch(syn_batch)

            # get half samples from history buffer
            half_from_hist = hist_buffer.get()
            # add to buffer
            hist_buffer.add(refined_batch)
            if len(half_from_hist) > 0:
                refined_batch[:BATCH_SIZE // 2] = half_from_hist

            D_loss_real = D.train_on_batch(real_batch, y_real)
            D_loss_ref = D.train_on_batch(refined_batch, y_refined)
            with tb_writer.as_default():
                tf.summary.scalar("D_real", D_loss_real, step=i)
                tf.summary.scalar("D_ref", D_loss_ref, step=i)
            print(f"D_real={D_loss_real} D_refined={D_loss_ref}")

        if (i % DEBUG_INTERVAL) == 0:
            syn_batch = next(syn_ds)
            with tb_writer.as_default():
                tf.summary.image("Adv refined",
                    (R.predict_on_batch(syn_batch[:N_PLOG_IMGS]) + 1.0) / 2.0,
                        max_outputs=N_PLOG_IMGS, step=i)
            D.save(path.join(cache_dir, f"D_{i}.keras"))
            R.save(path.join(cache_dir, f"R_{i}.keras"))
            combined.save(path.join(cache_dir, f"combined_{i}.keras"))

def main():
    args = get_cl_args()

    R = refiner_model(INPUT_SHAPE)
    D = discriminator_model(INPUT_SHAPE)

    R.compile(loss=self_regularization_loss, optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))
    D.compile(loss=local_adversarial_loss, optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))

    syn_inp = layers.Input(shape=INPUT_SHAPE)
    refined_out = R(syn_inp)
    refined_disc_score_out = D(refined_out)

    combined = Model(inputs=syn_inp, outputs=[refined_out, refined_disc_score_out], name="combined")
    D.trainable = False
    combined.compile(loss=[self_regularization_loss, local_adversarial_loss],
                     optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))

    real_ds = H5DatasetIter(args.real_ds_h5, BATCH_SIZE)
    syn_ds = H5DatasetIter(args.syn_ds_h5, BATCH_SIZE)

    if args.refiner_model:
        R.load_weights(args.refiner_model)
    else:
        pretrain_refiner(R, syn_ds)

    if args.discriminator_model:
        D.load_weights(args.discriminator_model)
    else:
        pretrain_discriminator(D, R, real_ds, syn_ds)

    adversarial_training(R, D, combined, syn_ds, real_ds)

if __name__ == "__main__":
    main()
