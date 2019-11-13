# For training image classification models

import os, datetime, json

import tensorflow as tf
import larq as lq

import click
from zookeeper import cli, build_train

from experiment import Experiment
import models, data, optimizers
import callbacks

# Register train command and associated switches
@cli.command()
@click.option("--name", default="Classify")
@click.option("--observer", default=None)
@click.option("--tb-graph", is_flag=True)
@build_train()
def train(build_model, dataset, hparams, logdir, name, observer, tb_graph):
    
    # Check if the given directory already contains model
    if os.path.exists(f"{logdir}/stats.json"):
        # then we will load the model weights
        model_dir = logdir
    else:
        # otherwise, create 
        # location to save the model -- <logdir>/<dataset>/<model>/<timestamp>
        model_dir = os.path.join(logdir,
                                dataset.dataset_name,
                                build_model.__name__,
                                datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "weights.h5")
    

    # Check if any observers are added for experiment.
    # If not, add a file_storage observer
    if observer is None:
        observer = f"file_storage={model_dir}"

    # Create an experiment
    ex = Experiment(name, 
                    dataset.dataset_name, 
                    build_model.__name__, 
                    hparams, 
                    observer)

    # Main function to run for experiment
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

        # Print Summary of models
        lq.models.summary(model)

        # If the model already exists, load it and continue training
        initial_epoch = 0
        if os.path.exists(os.path.join(model_dir, "stats.json")):
            with open(os.path.join(model_dir, "stats.json"), "r") as stats_file:
                initial_epoch = json.load(stats_file)["epoch"]
                click.echo(f"Restoring model {model_path} at epoch = {initial_epoch}")
                model.load_weights(model_path)


        cb = [
            callbacks.SaveStats(model_dir=model_dir)
        ]

        if tb_graph:
            # If tensorboard logging is enabled, write graph
            cb.extend(
                [
                    tf.keras.callbacks.TensorBoard(
                        log_dir=os.path.join(model_dir, "tb"),
                        write_graph=tb_graph,
                        histogram_freq=0,
                        update_freq='epoch',
                        # update_freq=0,
                        profile_batch=0,
                        embeddings_freq=0
                    ),
                ]
            )
        
        # Callback for sending data to Sacred Experiment
        cb.extend([
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs:
                    [ 
                        ex.log_scalar(metric, value, epoch + 1) for (metric, value) in logs.items()
                    ]
            )
        ])


        # Train this mode
        train_log = model.fit(
                        dataset.train_data(hparams.batch_size),
                        epochs=hparams.epochs,
                        steps_per_epoch=dataset.train_examples // hparams.batch_size,
                        validation_data=dataset.validation_data(hparams.batch_size),
                        validation_steps=dataset.validation_examples // hparams.batch_size,
                        initial_epoch=initial_epoch,
                        callbacks=cb
                    )
        
        # Save the model
        # not required as ModelCheckpoint already does this

    # Execute the experiment
    ex.execute()

# Register train command and associated switches
@cli.command()
@build_train()
def test(build_model, dataset, hparams, logdir):
    # Check if the given directory already contains model
    if os.path.exists(os.path.join(logdir, "stats.json")):
        # then mark this as the directory to load weights from
        model_dir = logdir
    else:
        # Raise Error
        raise RuntimeError("No valid model stats file found in directory", logdir)
		
    model_path = os.path.join(model_dir, "model")

    # Build model
    model = build_model(hparams, **dataset.preprocessing.kwargs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.make_optimizer(hparams.optimizer, hparams.opt_param),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    # Print Summary of models
    lq.models.summary(model)

    # # Load model weights from the specified file
    # model.load_weights(model_path)
    tf.train.Checkpoint(model=model).restore(model_path)

    # Test this model
    test_log = model.evaluate(
        dataset.test_data(hparams.batch_size),
        steps = dataset.test_examples // hparams.batch_size
    )

    print("+Test Results--------------------+")
    for (idx, metric) in enumerate(model.metrics_names):
        print("| {:20s}    {:6.4f} |".format(metric, test_log[idx]))
    print("+--------------------------------+")

if __name__ == "__main__":
    cli()