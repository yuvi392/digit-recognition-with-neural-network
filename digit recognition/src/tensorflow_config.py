"""Configure TensorFlow logging before importing Keras modules."""

import os

# Hide TensorFlow INFO and WARNING messages while preserving errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
