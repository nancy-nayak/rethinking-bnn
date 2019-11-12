# For training image classification models

from os import path

import tensorflow as tf
import larq as lq

import click
from zookeeper import cli, build_train

from experiment import Experiment
import models, data, optimizers

# Register train command and associated switches
@cli.command()
@click.option("--name", default="Classify")
@click.option("--observer", default=None)
@click.option("--tensorboard", default=True)
@build_train()
def train(build_model, dataset, hparams, logdir, name, observer, tensorboard):
    
    # location to save the model
    model_dir = path.join(logdir, dataset.dataset_name, build_model.__name__)
    print("Saving to ", model_dir)
    if observer is None:
        observer = f"file_storage={model_dir}"

    ex = Experiment(name, 
                    dataset.dataset_name, 
                    build_model.__name__, 
                    hparams, 
                    observer)

    @ex.main
    def train(_run):
        # Build model
        model = build_model(hparams, **dataset.preprocessing.kwargs)
        
        # Compile model
        model.compile(
            optimizer=optimizers.make_optimizer(hparams.optimizer, hparams.opt_param),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
        )

        lq.models.summary(model)

        train_log = model.fit(
                        dataset.train_data(hparams.batch_size),
                        epochs=hparams.epochs,
                        steps_per_epoch=dataset.train_examples // hparams.batch_size,
                        validation_data=dataset.validation_data(hparams.batch_size),
                        validation_steps=dataset.validation_examples // hparams.batch_size
                    )
        
        # Log the performace values to sacred experiment
        for (metric, values) in train_log.history.items():
            for (idx, value) in enumerate(values):
                _run.log_scalar(metric, value, idx)

    ex.execute()

if __name__ == "__main__":
    cli()