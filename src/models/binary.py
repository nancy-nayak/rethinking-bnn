# Archictectures for Binary Neural Networks

import tensorflow as tf
import larq as lq
from zookeeper import registry, HParams

# Generic Binary ConvNet 
@registry.register_model
def bcnn(hparams, input_shape, num_classes):
    kwargs = dict(
        input_quantizer="ste_sign",
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
    filters = [64, 32, 16]
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
@registry.register_hparams(bcnn)
class bvgg(default):
    # filters = [128, 128, 256, 256, 512, 512]
    filters = [128, 256, 256, 512]
    dense_units = [1024, 1024]


    