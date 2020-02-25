import tensorflow as tf
from keras import backend as K

from tensorflow_addons.metrics.utils import MeanMetricWrapper


def hamming_loss_fn(y_true, y_pred, threshold=0.5, mode="multilabel"):
    """Computes hamming loss.
    Hamming loss is the fraction of wrong labels to the total number
    of labels.
    In multi-class classification, hamming loss is calculated as the
    hamming distance between `actual` and `predictions`.
    In multi-label classification, hamming loss penalizes only the
    individual labels.
    Args:
        y_true: actual target value
        y_pred: predicted target value
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        mode: multi-class or multi-label
    Returns:
        hamming loss: float
    Usage:
    ```python
    # multi-class hamming loss
    hl = HammingLoss(mode='multiclass', threshold=0.6)
    hl.update_state(actuals, predictions)
    print('Hamming loss: ', hl.result().numpy()) # 0.25
    # multi-label hamming loss
    hl = HammingLoss(mode='multilabel', threshold=0.8)
    actuals = tf.constant([[1, 0, 1, 0],[0, 1, 0, 1],
                       [0, 0, 0,1]], dtype=tf.int32)
    predictions = tf.constant([[0.82, 0.5, 0.90, 0],
                               [0, 1, 0.4, 0.98],
                               [0.89, 0.79, 0, 0.3]],
                               dtype=tf.float32)
    hl.update_state(actuals, predictions)
    print('Hamming loss: ', hl.result().numpy()) # 0.16666667
    ```
    """
    if mode not in ['multiclass', 'multilabel']:
        raise TypeError('mode must be either multiclass or multilabel]')

    if threshold is None:
        threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
        # make sure [0, 0, 0] doesn't become [1, 1, 1]
        # Use abs(x) > eps, instead of x != 0 to check for zero
        y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
    else:
        y_pred = y_pred > threshold

    #     y_true = tf.cast(y_true, tf.int32)
    #     y_pred = tf.cast(y_pred, tf.int32)

    if mode == 'multiclass':
        nonzero = tf.cast(
            K.sum(K.abs(y_true * y_pred), axis=-1), tf.float32)
        return 1.0 - nonzero

    else:
        nonzero = tf.cast(
            K.sum(K.abs(y_true - y_pred), axis=-1), tf.float32)
        return nonzero / tf.cast(y_pred.get_shape()[-1], tf.float32)


class HammingLoss(MeanMetricWrapper):
    """Computes hamming loss."""

    def __init__(self,
                 mode,
                 name='hamming_loss',
                 threshold=None,
                 dtype=tf.float32):
        super(HammingLoss, self).__init__(
            hamming_loss_fn, name, dtype=dtype, mode=mode, threshold=threshold)


def hamming_loss(y_true, y_pred):
    incorrects = K.cast(K.sum(K.abs(y_true - y_pred), axis=-1), dtype=tf.float32)
    means = K.mean(incorrects, axis=-1)
    return means
