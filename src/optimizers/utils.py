import tensorflow as tf
import larq as lq

# Utility function to make optimizer from hyper param settings
def make_optimizer(optimizer, opt_param):
    # Make optimizer from the given settings
    optimizer_lookup = {
        "Adam"      : tf.keras.optimizers.Adam,
        "SGD"       : tf.keras.optimizers.SGD,
        "RMSprop"   : tf.keras.optimizers.RMSprop
    }

    if optimizer in optimizer_lookup:
        # This is one of the traditional optimizers
        return optimizer_lookup[optimizer](**opt_param)
    elif optimizer == "Bop":
        return lq.optimizers.CaseOptimizer(
                    (
                        lq.optimizers.Bop.is_binary_variable,
                        lq.optimizers.Bop(**opt_param),
                    ),
                    default_optimizer=tf.keras.optimizers.Adam(0.01),  # for FP weights
                )
    else:
        raise RuntimeError(f"Optimizer '{optimizer} is not implemented.")