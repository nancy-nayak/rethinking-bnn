# Callbacks for training

import os, json
import tensorflow as tf

class SaveStats(tf.keras.callbacks.LambdaCallback):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.stats_file = os.path.join(model_dir, "stats.json")
        self.weights_file = os.path.join(model_dir, "weights.h5")
        
        def save_stats(epoch, logs=None):
            self.model.save_weights(self.weights_file)
            with open(self.stats_file, "w") as f:
                json.dump({"epoch": epoch + 1}, f)
        
        super().__init__(on_epoch_end=save_stats)
