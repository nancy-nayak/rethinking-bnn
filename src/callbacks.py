# Callbacks for training

import os, json
import tensorflow as tf

class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, experiment, *args, **kwargs):
        self.experiment = experiment
        super().__init__(*args, **kwargs)
    
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        with open(os.path.join(os.path.dirname(self.filepath), "stats.json"), "w") as stats_file:
            # Save current epoch
            json.dump({"epoch": epoch + 1}, stats_file)
        # Update values to scared experiment 
        for (metric, value) in logs.items():
            self.experiment.log_scalar(metric, value, epoch + 1)
        