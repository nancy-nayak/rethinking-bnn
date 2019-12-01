# Preprocessing for different tasks

from zookeeper import Preprocessing
import tensorflow as tf

class ImageClassification(Preprocessing):
    @property
    def kwargs(self):
        return {
            "input_shape": self.features["image"].shape,
            "num_classes": self.features["label"].num_classes
        }

    def inputs(self, data):
        return tf.cast(data["image"], tf.float32)

    def outputs(self, data):
        return tf.one_hot(data["label"], self.features["label"].num_classes)

class AutoEncoding(Preprocessing):
    @property
    def kwargs(self):
        return{
            "image_shape": self.features["image"].shape
        }
    
    def inputs(self, data):
        return tf.cast(data["image"], tf.float32)
    
    def outputs(self, data):
        return tf.cast(data["image"], tf.float32)