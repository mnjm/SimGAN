import tensorflow as tf
from tensorflow.keras import optimizers, layers, Model
from os import path, mkdir
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import DatasetIter, HistoryBuffer
from models import refiner_model, descriminator_model
from datetime import datetime

INPUT_SHAPE = (35, 55, 1)

# INFO: Hyper params
LEARNING_RATE = 0.001
LAMBDA = 0.0002 # Self Regularization Loss lambda param FIXME: find out approp value for this
BATCH_SIZE = 512
HIST_BUFFER_SIZE = 100 * BATCH_SIZE # FIXME: Approp size for HistoryBuffer ?
STEPS = 10000
D_UPDATE_PER_STEP = 1
R_UPDATE_PER_STEP = 2

LOG_INTERVAL = 100

def self_regularization_loss(y_t, y_p, lambda_):
    return tf.multiply(lambda_, tf.reduce_sum(tf.abs(y_p - y_t)))

def local_adversarial_loss(y_t, y_p):
    y_t = tf.reshape(y_t, (-1 ,2))
    y_p = tf.reshape(y_p, (-1 ,2))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_t, predicted=y_p))

R = refiner_model(INPUT_SHAPE)
D = descriminator_model(INPUT_SHAPE)

self_reg_loss = lambda yt,yp: self_regularization_loss(yt, yp, tf.Constant(LAMBDA))
R.compile(loss=self_reg_loss, optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))

D.compile(loss=local_adversarial_loss, optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))

syn_inp = layers.Input(shape=INPUT_SHAPE)
refined_out = R(syn_inp)
refined_desc_score_out = D(refined_out)

combined = Model(inputs=syn_inp, outputs=[refined_out, refined_desc_score_out], name="combined")
D.trainable = False
combined.compile(loss=[self_reg_loss, local_adversarial_loss], optimizer=optimizers.SGD(learning_rate=LEARNING_RATE))
