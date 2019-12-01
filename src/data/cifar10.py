# CIFAR10 dataset

import tensorflow as tf
from zookeeper import registry
from .tasks import ImageClassification, AutoEncoding


@registry.register_preprocess("cifar10")
class default(ImageClassification):
    def inputs(self, data, training):
        image = data["image"]
        if training:
            # Resize and flip
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            image = tf.image.random_crop(image, [32, 32, 3])
            image = tf.image.random_flip_left_right(image)

        return tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0

@registry.register_preprocess("cifar10")
class AE(AutoEncoding):
    def inputs(self, data, training):
        image = data["image"]
        noise_factor = 0.1
        return tf.clip_by_value((tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0 ) + noise_factor* tf.random.normal(shape=image.shape, mean=0.0, stddev=1.0), -1.0, 1.0)

    def outputs(self, data, training):
        image = data["image"]
        noise_factor = 0.1
        return tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0 