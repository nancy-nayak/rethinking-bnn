# Basic models

import tensorflow as tf
from zookeeper import registry, HParams

# Simple fully connected network
@registry.register_model
def fcn(hparams, input_shape, num_classes):

    # 1. Create Input layer
    input_layer = tf.keras.layers.Input(shape=input_shape)
    # 2. Input is an image, flatten it out to use with Dense layers
    x = tf.keras.layers.Flatten()(input_layer)
    # 3. Create hidden layers with hidden units specified in `hparams`
    for units in hparams.hidden_units:
        x = tf.keras.layers.Dense(units, activation=hparams.activation)(x)
    # 4. Make output layer
    output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Create model and return
    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

@registry.register_hparams(fcn)
class default(HParams):
    # Network architecture related hyper parameters
    activation = "relu"
    hidden_units = [64, 64]

    # Training related hyperparameters
    epochs = 5
    batch_size = 32

    # Optimizer ...
    optimizer = "Adam"
    # ... followed by the parameters
    opt_param = dict(
        lr = 1e-3,
        beta_1 = 0.99,
        beta_2 = 0.999
    )


# Simple Convolutional network
@registry.register_model
def cnn(hparams, input_shape, num_classes):

    in_layer = tf.keras.layers.Input(shape=input_shape)
    
    # Make convolutional layers followed by pooling
    x = in_layer
    for n_filters in hparams.filters:
        x = tf.keras.layers.Conv2D(filters=n_filters, 
                             kernel_size=hparams.kernel_size,
                             activation=hparams.activation)(x)
        x = tf.keras.layers.AveragePooling2D()(x)
    # Make dense layer
    x = tf.keras.layers.Flatten()(x)
    for units in hparams.hidden_units:
        x = tf.keras.layers.Dense(units=units, activation=hparams.activation)(x)
    
    # Make output
    out_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Return model
    return tf.keras.models.Model(inputs=in_layer, outputs=out_layer)

@registry.register_hparams(cnn)
class default(HParams):
    activation = "relu"
    filters = [4, 4]
    kernel_size = (3, 3)
    hidden_units = [64, 64]

    epochs = 10
    batch_size = 32

    optimizer = "Adam"
    opt_param = dict(
        lr = 1e-3,
        beta_1 = 0.99,
        beta_2 = 0.999
    )

@registry.register_hparams(cnn)
class lenet5(default):
    """
        Architecture of classic LeNet-5
        Source: https://engmrk.com/lenet-5-a-classic-cnn-architecture/
    """
    filters = [6, 16]
    kernel_size = (5, 5)
    hidden_units = [120, 84]




