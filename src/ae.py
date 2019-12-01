# For training Auto Encoder models

import os, datetime, json

import tensorflow as tf
import larq as lq
import tensorflow.keras.backend as K
import math

import click
from zookeeper import cli, build_train

from experiment import Experiment
import models, data, optimizers
import callbacks

import matplotlib.pyplot as plt

# Register train command and associated switches
@cli.command()
@click.option("--name", default="AE")
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
        
        
        # Custom metric
        import math
        def PSNR(y_true, y_pred):
            # print(K.mean(K.square(y_pred - y_true)))
            # exit()
            max_pixel = 1.0
            return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
        
        # Compile model
        model.compile(
            optimizer=optimizers.make_optimizer(hparams.optimizer, hparams.opt_param),
            loss="mse",
            metrics=[PSNR]
        )

        # Print Summary of models
        lq.models.summary(model)

        # If the model already exists, load it and continue training
        initial_epoch = 0
        if os.path.exists(os.path.join(model_dir, "stats.json")):
            with open(os.path.join(model_dir, "stats.json"), "r") as stats_file:
                initial_epoch = json.load(stats_file)["epoch"]
                click.echo(f"Restoring model from {model_path} at epoch = {initial_epoch}")
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

        train_log = model.fit(
                        dataset.train_data(hparams.batch_size),
                        epochs=hparams.epochs,
                        steps_per_epoch=dataset.train_examples // hparams.batch_size,
                        validation_data=dataset.validation_data(hparams.batch_size),
                        validation_steps=dataset.validation_examples // hparams.batch_size,
                        initial_epoch=initial_epoch,
                        callbacks=cb
                    )

    # Execute the experiment
    ex.execute()

# Register test command and associated switches
@cli.command()
@build_train()
def test(build_model, dataset, hparams, logdir):
    # Check if the given directory already contains model
    if os.path.exists(os.path.join(logdir, "stats.json")):
        # then mark this as the directory to load weights from
        model_dir = logdir
    else:
        # Raise Error
        raise RuntimeError(f"No valid model stats file found in {logdir}")	
    model_path = os.path.join(model_dir, "weights.h5")

    # Build model
    model = build_model(hparams, **dataset.preprocessing.kwargs)
    
    # Custom metric
    import math
    def PSNR(y_true, y_pred):
        max_pixel = 1.0
        return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
        
    # Compile model
    model.compile(
        optimizer=optimizers.make_optimizer(hparams.optimizer, hparams.opt_param),
        loss="mse",
        metrics=[PSNR]
    )

    # Print Summary of models
    lq.models.summary(model)

    # # Load model weights from the specified file
    model.load_weights(model_path)
    
    # Test this model
    test_log = model.evaluate(
        dataset.test_data(hparams.batch_size),
        steps = dataset.test_examples // hparams.batch_size
    )

    data = [["Metric", "Value"]]
    for (idx, metric) in enumerate(model.metrics_names):
        data.append([metric, test_log[idx]])
    
    from terminaltables import AsciiTable
    print(AsciiTable(data, title="Test Statistics").table)



# Register visualization of pictures command 
@cli.command()
@build_train()
def vispics(build_model, dataset, hparams, logdir):
    # Check if the given directory already contains model
    if os.path.exists(os.path.join(logdir, "stats.json")):
        # then mark this as the directory to load weights from
        model_dir = logdir
    else:
        # Raise Error
        raise RuntimeError(f"No valid model stats file found in {logdir}")	
    model_path = os.path.join(model_dir, "weights.h5")

    # Build model
    model = build_model(hparams, **dataset.preprocessing.kwargs)

    # Custom metric
    import math
    def PSNR(y_true, y_pred):
        max_pixel = 1.0
        return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
            
    # Compile model
    model.compile(
        optimizer=optimizers.make_optimizer(hparams.optimizer, hparams.opt_param),
        loss="mse",
        metrics=[PSNR]
    )

    # # Print Summary of models
    # lq.models.summary(model)

    viz_model = tf.keras.models.Model(inputs=model.input,
                                        outputs=[model.input, model.output])
    
    # Load model weights from the specified file
    model.load_weights(model_path)

    n_samples = 10000
    recon_imgs = viz_model.predict(dataset.test_data(n_samples), steps = 1)

    import matplotlib.pyplot as plt
    import numpy as np



    ### Uncomment this if u want Comparison of reconstructed images
    ### using BAE+BOP with actual images =================================================================
    
    PSNR = np.zeros(n_samples)
    for idx in range(n_samples):
        print(idx)
        loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx,:,:,:], axis=0))
        PSNR[idx] = psnr

    idxsorted = np.argsort(PSNR)
    
    idx0 = idxsorted[0]
    idx1 = idxsorted[1]
    idx4999 = idxsorted[int(n_samples/2-1)]
    idx9998 = idxsorted[n_samples-2]
    idx9999 = idxsorted[n_samples-1]
    print(idx0, idx1, idx4999, idx9998, idx9999)

    fig, ax = plt.subplots(2,5, figsize=(7, 3.5))  
    # Plot the 5 samples 
    ax[0,0].imshow(0.50 * (recon_imgs[0][idx0,:,:,:]+1))
    ax[0,0].set_ylabel('Noisy image')
    ax[1,0].imshow(0.50 * (recon_imgs[1][idx0,:,:,:]+1))
    ax[1,0].set_xlabel('PSNR = {0:.2f}'.format(PSNR[idx0]))
    ax[1,0].set_ylabel('BAE w/ BOP')

    ax[0,1].imshow(0.50 * (recon_imgs[0][idx1,:,:,:]+1))
    ax[1,1].imshow(0.50 * (recon_imgs[1][idx1,:,:,:]+1))
    ax[1,1].set_xlabel('PSNR = {0:.2f}'.format(PSNR[idx1]))

    ax[0,2].imshow(0.50 * (recon_imgs[0][idx4999,:,:,:]+1))
    ax[1,2].imshow(0.50 * (recon_imgs[1][idx4999,:,:,:]+1))
    ax[1,2].set_xlabel('PSNR = {0:.2f}'.format(PSNR[idx4999]))

    ax[0,3].imshow(0.50 * (recon_imgs[0][idx9998,:,:,:]+1))
    ax[1,3].imshow(0.50 * (recon_imgs[1][idx9998,:,:,:]+1))
    ax[1,3].set_xlabel('PSNR = {0:.2f}'.format(PSNR[idx9998]))

    ax[0,4].imshow(0.50 * (recon_imgs[0][idx9999,:,:,:]+1))
    ax[1,4].imshow(0.50 * (recon_imgs[1][idx9999,:,:,:]+1))
    ax[1,4].set_xlabel('PSNR = {0:.2f}'.format(PSNR[idx9999]))

    fig.savefig("./../results/BAEBOPfigsbestworst.pdf", format = 'pdf', bbox_inches = 'tight' )
    plt.show()

    exit()

    ## Once u find the five indeces use them to find the reconstructed figures with AE+Adam and BAE+Adam
    idx0 = 2590
    idx1 = 6869
    idx4999 = 4728
    idx9998 = 8264
    idx9999 = 9701

    # #### Uncomment this if u want BAE+Adam =====================================================
    # fig, ax = plt.subplots(1,5, figsize=(7, 2))  
    # # Test this model

    # loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx0,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx0,:,:,:], axis=0))
    # ax[0].imshow(0.50 * (recon_imgs[1][idx0,:,:,:]+1))
    # ax[0].set_xlabel('PSNR = {0:.2f}'.format(psnr))
    # ax[0].set_ylabel('BAE w/ Adam')

    # loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx1,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx1,:,:,:], axis=0))
    # ax[1].imshow(0.50 * (recon_imgs[1][idx1,:,:,:]+1))
    # ax[1].set_xlabel('PSNR = {0:.2f}'.format(psnr))

    # loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx4999,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx4999,:,:,:], axis=0))
    # ax[2].imshow(0.50 * (recon_imgs[1][idx4999,:,:,:]+1))
    # ax[2].set_xlabel('PSNR = {0:.2f}'.format(psnr))

    # loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx9998,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx9998,:,:,:], axis=0))
    # ax[3].imshow(0.50 * (recon_imgs[1][idx9998,:,:,:]+1))
    # ax[3].set_xlabel('PSNR = {0:.2f}'.format(psnr))

    # loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx9999,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx9999,:,:,:], axis=0))
    # ax[4].imshow(0.50 * (recon_imgs[1][idx9999,:,:,:]+1))
    # ax[4].set_xlabel('PSNR = {0:.2f}'.format(psnr))

    # fig.savefig("./../results/BAEAdamfigsbestworst.pdf", format = 'pdf', bbox_inches = 'tight' )
    # plt.show()

    # #### Uncomment this if u want BAE+BOP =============================================================
    # fig, ax = plt.subplots(1,5, figsize=(7, 2))  
   
    # loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx0,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx0,:,:,:], axis=0))
    # ax[0].imshow(0.50 * (recon_imgs[1][idx0,:,:,:]+1))
    # ax[0].set_xlabel('PSNR = {0:.2f}'.format(psnr))
    # ax[0].set_ylabel('AE w/ Adam')

    # loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx1,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx1,:,:,:], axis=0))
    # ax[1].imshow(0.50 * (recon_imgs[1][idx1,:,:,:]+1))
    # ax[1].set_xlabel('PSNR = {0:.2f}'.format(psnr))

    # loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx4999,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx4999,:,:,:], axis=0))
    # ax[2].imshow(0.50 * (recon_imgs[1][idx4999,:,:,:]+1))
    # ax[2].set_xlabel('PSNR = {0:.2f}'.format(psnr))

    # loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx9998,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx9998,:,:,:], axis=0))
    # ax[3].imshow(0.50 * (recon_imgs[1][idx9998,:,:,:]+1))
    # ax[3].set_xlabel('PSNR = {0:.2f}'.format(psnr))

    # loss, psnr = model.evaluate(np.expand_dims(recon_imgs[1][idx9999,:,:,:], axis=0), np.expand_dims(recon_imgs[0][idx9999,:,:,:], axis=0))
    # ax[4].imshow(0.50 * (recon_imgs[1][idx9999,:,:,:]+1))
    # ax[4].set_xlabel('PSNR = {0:.2f}'.format(psnr))

    # fig.savefig("./../results/AEAdamfigsbestworst.pdf", format = 'pdf', bbox_inches = 'tight' )
    # plt.show()
    

if __name__ == "__main__":
	cli()