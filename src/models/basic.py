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

@registry.register_hparams(fcn)
class l5(default):
    hidden_units = [128, 128, 64, 64, 32]

# Simple fully connected network
@registry.register_model
def fcn5(hparams, input_shape, num_classes):

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

@registry.register_hparams(fcn5)
class default(HParams):
    # Network architecture related hyper parameters
    activation = "relu"
    hidden_units = [128, 64, 64, 32]

    # Training related hyperparameters
    epochs = 5
    batch_size = 32

    # Optimizer and associated hyperparameters
    optimizer = "Adam" # Use Keras string identifier here
    # ... followed by the parameters
    lr = 1e-3
    beta_1 = 0.99
    beta_2 = 0.999
    