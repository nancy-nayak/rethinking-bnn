# Archictectures for Binary Neural Networks

import tensorflow as tf
import larq as lq
from zookeeper import registry, HParams

@registry.register_model
def binaryae(hparams, image_shape):

    input_shape = image_shape

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
    x = lq.layers.QuantConv2D(32, (3, 3), padding="same",
                                kernel_quantizer=hparams.kernel_quantizer,
                                kernel_constraint=hparams.kernel_constraint,
                                use_bias=False,
                                input_shape=input_shape)(in_layer)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = lq.layers.QuantConv2D(16, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = lq.layers.QuantConv2D(8, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    encoded = x
    
    x = lq.layers.QuantConv2DTranspose(8, (3, 3), padding="same", kernel_quantizer=hparams.kernel_quantizer,
                                                           kernel_constraint=hparams.kernel_constraint)(encoded)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    x = lq.layers.QuantConv2DTranspose(16, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    x = lq.layers.QuantConv2DTranspose(32, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    x = lq.layers.QuantConv2DTranspose(3, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.Activation("tanh")(x)

    out = x
   
    return tf.keras.models.Model(inputs=in_layer, outputs=out)

@registry.register_hparams(binaryae)
class bnn(HParams):
    input_quantizer = "ste_sign"
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"

    # Normalization 
    norm_layer = "BatchNormalization"
    norm_param = dict(
        scale=False
    )
    
    # Training properties
    epochs = 100
    batch_size = 64

    optimizer = "Adam"
    opt_param = dict(
        lr = 1e-3,
        beta_1 = 0.99,
        beta_2 = 0.999
    )

@registry.register_hparams(binaryae)
class bop(bnn):
    optimizer = "Bop"
    opt_param = dict(
        threshold=1e-6,
        gamma=1e-3
    )  


@registry.register_model
def autoencoder(hparams, image_shape):
    conv_layer = tf.keras.layers.Conv2D
    deconv_layer = tf.keras.layers.Conv2DTranspose
    # normalization_layer = tf.keras.layers.BatchNormalization

    # Decide which normalization to use
    norm_lookup = {
        "BatchNormalization": tf.keras.layers.BatchNormalization,
        "LayerNormalization": tf.keras.layers.LayerNormalization
    }
    if not hparams.norm_layer in norm_lookup:
        raise NotImplementedError("Unknown normalization layer {}".format(hparams.norm_layer))
    # We know this normalization layer
    normalization_layer = norm_lookup[hparams.norm_layer]

    input_shape = image_shape

    in_layer = tf.keras.layers.Input(shape=input_shape)

    x = conv_layer(32, (3, 3), activation="relu", padding="same")(in_layer)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_layer(16, (3, 3), activation="relu",padding="same")(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv_layer(8, (3, 3), activation="relu",padding="same")(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    encoded = x

    x = deconv_layer(8, (3, 3), activation="relu", padding="same")(encoded)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    x = deconv_layer(16, (3, 3), activation="relu", padding="same")(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    x = deconv_layer(32, (3, 3), activation="relu", padding="same")(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    x = deconv_layer(3, (3, 3), activation="tanh", padding="same")(x)

    out = x

    return tf.keras.models.Model(inputs=in_layer, outputs=out)

@registry.register_hparams(autoencoder)
class traditional(HParams):
    # Normalization 
    norm_layer = "BatchNormalization"
    norm_param = dict(
        scale=False
    )
  # Training properties
    epochs = 100
    batch_size = 64

    optimizer = "Adam"
    opt_param = dict(
        lr = 1e-3,
        beta_1 = 0.99,
        beta_2 = 0.999
    )
