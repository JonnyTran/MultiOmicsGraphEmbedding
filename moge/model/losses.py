import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_addons.metrics.utils import MeanMetricWrapper


def hamming_loss_fn(y_true, y_pred, threshold=0.5, mode="multilabel"):
    if mode not in ['multiclass', 'multilabel']:
        raise TypeError('mode must be either multiclass or multilabel]')

    if threshold is None:
        threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
        # make sure [0, 0, 0] doesn't become [1, 1, 1]
        # Use abs(x) > eps, instead of x != 0 to check for zero
        y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
    else:
        y_pred = y_pred > threshold

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    if mode == 'multiclass':
        nonzero = tf.cast(
            K.sum(tf.abs(y_true * y_pred), axis=-1), tf.float32)
        return 1.0 - nonzero

    else:
        nonzero = tf.cast(
            K.sum(tf.abs(y_true - y_pred), axis=-1), tf.float32)
        return K.mean(nonzero / tf.cast(y_pred.get_shape()[-1], tf.float32))


class HammingLoss(MeanMetricWrapper):
    def __init__(self, mode, name='hamming_loss', threshold=None, dtype=tf.float32):
        super(HammingLoss, self).__init__(
            hamming_loss_fn, name, dtype=dtype, mode=mode, threshold=threshold)


def hamming_loss(y_true, y_pred):
    incorrects = K.cast(K.sum(K.abs(y_true - y_pred), axis=-1), dtype=tf.float32)
    means = K.mean(incorrects, axis=-1)
    return means


def coverage_error(y_true, y_pred):
    max_predicted = K.max(y_true * y_pred, axis=-1)
    return K.mean(max_predicted - 1, axis=-1)


def one_error(y_true, y_pred):
    min_predicted = K.min(y_pred * y_true, axis=-1)
    print("min_predicted", min_predicted)
    max_nonlabels = K.max((1 - y_true) * y_pred, axis=-1)
    print("max_nonlabels", max_nonlabels)
    print("min_predicted + max_nonpredictions", min_predicted + max_nonlabels)
    return K.mean(min_predicted + max_nonlabels, axis=-1)


def focal_loss(y_true, y_pred, gamma=2, alpha=1.0):
    cross_entropy_loss = K.binary_crossentropy(y_true, y_pred, from_logits=False)
    print("cross_entropy_loss", cross_entropy_loss)
    p_t = ((y_true * y_pred) +
           ((1 - y_true) * (1 - y_pred)))
    modulating_factor = 1.0
    if gamma:
        modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = (y_true * alpha +
                               (1 - y_true) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                cross_entropy_loss)
    print("focal_cross_entropy_loss", focal_cross_entropy_loss)
    return K.mean(focal_cross_entropy_loss, axis=-1)