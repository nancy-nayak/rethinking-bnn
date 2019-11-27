# Archictectures for Binary Neural Networks

import tensorflow as tf
import larq as lq
from zookeeper import registry, HParams

# Generic Binary ConvNet 
@registry.register_model
def bcnn(hparams, input_shape, num_classes):
    kwargs = dict(
        input_quantizer=hparams.input_quantizer,
        kernel_quantizer=hparams.kernel_quantizer,
        kernel_constraint=hparams.kernel_constraint
    )

    in_layer = tf.keras.layers.Input(shape=input_shape)

    # First Conv Layer, 
    # Don't quantize input layer values
    x = lq.layers.QuantConv2D(hparams.filters[0], hparams.kernel_size,
                                kernel_quantizer=hparams.kernel_quantizer,
                                kernel_constraint=hparams.kernel_constraint,
                                use_bias=False,
                            )(in_layer)
    x = tf.keras.layers.BatchNormalization(scale=hparams.norm_scale)(x)

    # ConvLayers
    if len(hparams.filters) > 1:
        for n_filters in hparams.filters[1:]:
            x = lq.layers.QuantConv2D(n_filters, hparams.kernel_size,
                                        padding="same", **kwargs)(x)
            x = tf.keras.layers.MaxPool2D(pool_size=hparams.pool_size,
                                        strides=hparams.pool_stride)(x)
            x = tf.keras.layers.BatchNormalization(scale=hparams.norm_scale)(x)
    
    # Flatten
    x = tf.keras.layers.Flatten()(x)

    # Dense layers
    if len(hparams.dense_units) > 0:
        for units in hparams.dense_units:
            x = lq.layers.QuantDense(units, **kwargs)(x)
            x = tf.keras.layers.BatchNormalization(scale=hparams.norm_scale)(x)
    
    # Output layer
    x = lq.layers.QuantDense(num_classes, **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(scale=hparams.norm_scale)(x)
    out_layer = tf.keras.layers.Activation("softmax")(x)

    # Make model and return
    return tf.keras.models.Model(inputs=in_layer, outputs=out_layer)

@registry.register_hparams(bcnn)
class default(HParams):
    input_quantizer = "ste_sign"
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"
    
    # Conv layer properties
    filters = [128, 256, 512]
    kernel_size = (3,3)

    # Pool layer properties
    pool_size = (2, 2)
    pool_stride = (2, 2)

    # Dense layer properties
    dense_units = [256, 128, 64]

    # BatchNorm properties
    norm_scale = False
    
    # Training properties
    epochs = 100
    batch_size = 64

    optimizer = "Adam"
    opt_param = dict(
        lr = 1e-3,
        beta_1 = 0.99,
        beta_2 = 0.999
    )

# Binarized VGG model
@registry.register_model
def binarynet(hparams, input_shape, num_classes):
    kwargs = dict(
        input_quantizer=hparams.input_quantizer,
        kernel_quantizer=hparams.kernel_quantizer,
        kernel_constraint=hparams.kernel_constraint
    )

    # Input layer
    in_layer = tf.keras.layers.Input(shape=input_shape)
    
    # First hidden layer
    x = lq.layers.QuantConv2D(128, (3, 3),
                                kernel_quantizer=hparams.kernel_quantizer,
                                kernel_constraint=hparams.kernel_constraint,
                                use_bias=False,
                                input_shape=input_shape)(in_layer)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = lq.layers.QuantConv2D(128, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    x = lq.layers.QuantConv2D(256, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = lq.layers.QuantConv2D(256, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    x = lq.layers.QuantConv2D(512, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = lq.layers.QuantConv2D(512, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    x = tf.keras.layers.Flatten()(x)

    x = lq.layers.QuantDense(1024, **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    x = lq.layers.QuantDense(1024, **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    x = lq.layers.QuantDense(num_classes, **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    out_layer = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.models.Model(inputs=in_layer, outputs=out_layer)

@registry.register_hparams(binarynet)
class default(HParams):
    input_quantizer = "ste_sign"
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"
    
    # Training properties
    epochs = 100
    batch_size = 64

    optimizer = "Adam"
    opt_param = dict(
        lr = 1e-3,
        beta_1 = 0.99,
        beta_2 = 0.999
    )


@registry.register_hparams(binarynet)
class latentweights(HParams):
    input_quantizer = "ste_sign"
    kernel_quantizer = None
    kernel_constraint = None
    
    # Training properties
    epochs = 100
    batch_size = 64

    optimizer = "Adam"
    opt_param = dict(
        lr = 1e-3,
        beta_1 = 0.99,
        beta_2 = 0.999
    )


@registry.register_hparams(binarynet)
class bop(default):
    input_quantizer = "ste_sign"
    kernel_quantizer = "ste_sign"
    kernel_constraint = "weight_clip"

    threshold = 1e-6
    gamma = 1e-3

    optimizer = "Bop"
    opt_param = dict(
        threshold=threshold,
        gamma=gamma
    )


@registry.register_model
def binaryvgg(hparams, input_shape, num_classes):
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
    x = lq.layers.QuantConv2D(128, (3, 3),
                                kernel_quantizer=hparams.kernel_quantizer,
                                kernel_constraint=hparams.kernel_constraint,
                                use_bias=False,
                                input_shape=input_shape)(in_layer)
    x = normalization_layer(**hparams.norm_param)(x)
    x = lq.layers.QuantConv2D(128, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = normalization_layer(**hparams.norm_param)(x)

    x = lq.layers.QuantConv2D(256, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = lq.layers.QuantConv2D(256, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = normalization_layer(**hparams.norm_param)(x)

    x = lq.layers.QuantConv2D(512, (3, 3), padding="same", **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    x = lq.layers.QuantConv2D(512, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = normalization_layer(**hparams.norm_param)(x)

    x = tf.keras.layers.Flatten()(x)

    x = lq.layers.QuantDense(1024, **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)

    x = lq.layers.QuantDense(1024, **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)

    x = lq.layers.QuantDense(num_classes, **kwargs)(x)
    x = normalization_layer(**hparams.norm_param)(x)
    out_layer = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.models.Model(inputs=in_layer, outputs=out_layer)

@registry.register_hparams(binaryvgg)
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

@registry.register_hparams(binaryvgg)
class bop(default):
    optimizer = "Bop"
    opt_param = dict(
        threshold=1e-6,
        gamma=1e-3
    )