# CIFAR10 dataset

import tensorflow as tf
from zookeeper import registry
from .tasks import ImageClassification, AutoEncoding
import numpy as np 

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
        noise_factor = 0.5
        # if training:
        #     # Resize and flip
        image = image + noise_factor* np.random.normal(loc=0.0, scale=1.0, size=image.shape)
            
        return tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0