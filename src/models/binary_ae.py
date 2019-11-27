# Archictectures for Binary Neural Networks

import tensorflow as tf
import larq as lq
from zookeeper import registry, HParams

@registry.register_model
def binaryae(hparams, input_shape, num_classes):
    kwargs = dict(
        input_quantizer=hparams.input_quantizer,
        kernel_quantizer=hparams.kernel_quantizer,
        kernel_constraint=hparams.kernel_constraint
    )

    # Decide which normalization to use
    norm_lookup = {
        "BatchNormalization": tf.keras.layers.BatchNormalization,
        "LayerNormalization": tf.keras.layers.LayerNormalization
    }
    if not hparams.norm_layer in norm_lookup:
        raise NotImplementedError("Unknown normalization layer {}".format(hparams.norm_layer))
    # We know this normalization layer
    normalization_layer = norm_lookup[hparams.norm_layer]

    # Input layer
    in_layer = tf.keras.layers.Input(shape=input_shape)

    # First hidden layer
    x = lq.layers.QuantConv2D(64, (3, 3),
                                kernel_quantizer=hparams.kernel_quantizer,
                                kernel_constraint=hparams.kernel_constraint,
                                use_bias=False,
                                input_shape=input_shape)(in_layer)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = lq.layers.QuantConv2D(32, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = lq.layers.QuantConv2D(16, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    encoded = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    

    x = lq.layers.QuantConv2D(16, (3, 3), padding="same", **kwargs)(encoded)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    x = lq.layers.QuantConv2D(32, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    x = lq.layers.QuantConv2D(64, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    x = lq.layers.QuantConv2D(3, (4, 4), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    decoded = tf.keras.layers.Activation("sigmoid")(x)
    print("decoded shape:{}".format(decoded.shape))
    exit()
    return tf.keras.models.Model(inputs=in_layer, outputs=decoded)

@registry.register_hparams(binaryae)
class default(HParams):
    input_quantizer = "ste_sign"
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"

    # Normalization 
    norm_layer = "BatchNormalization"
    norm_param = dict(
        scale=False
    )
    
    # Training properties
    epochs = 250
    batch_size = 64

    optimizer = "Adam"
    opt_param = dict(
        lr = 1e-3,
        beta_1 = 0.99,
        beta_2 = 0.999
    )

@registry.register_hparams(binaryae)
class bop(default):
    optimizer = "Bop"
    opt_param = dict(
        threshold=1e-6,
        gamma=1e-3
    )  
