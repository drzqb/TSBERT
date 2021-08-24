import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader


def create_initializer(stddev=0.02):
    return TruncatedNormal(stddev=stddev)


def softmax(a, mask):
    """
    :param a: B*ML1*ML2
    :param mask: B*ML1*ML2
    """
    return tf.nn.softmax(tf.where(mask, a, (1. - tf.pow(2., 31.)) * tf.ones_like(a)), axis=-1)


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def get_cos_distance(X1, X2):
    # calculate cos distance between two sets
    # more similar more big
    # 求模
    X1_norm = tf.nn.l2_normalize(X1, axis=-1)
    X2_norm = tf.nn.l2_normalize(X2, axis=-1)
    # 内积余弦
    return tf.reduce_sum(X1_norm * X2_norm, axis=-1)
