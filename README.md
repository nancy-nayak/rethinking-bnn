# Rethinking-bnn
Repository for NIPS Reproducibility Challenge 2019 for the paper titled "Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization" and the report is named "A comprehensive study on binary optimizer and its applicability", can be found [here](https://openreview.net/pdf?id=y53aaSM5o).


Thanks to [CodeOcean](https://codeocean.com/) and [Google colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) for compute support.

# Requirements
- [Python](https://python.org) version `3.6` or `3.7`
- [Tensorflow](https://www.tensorflow.org/install) version `2.0.0`
- [Larq](https://github.com/plumerai/larq) version `0.8.2`
- [Zookeeper](https://github.com/plumerai/zookeeper) version `0.5.5`
- [Sacred](https://github.com/IDSIA/sacred) version `0.8.0`


# Reproduce paper experiments

## Ablation studies for threshold and gamma for Batchnorm and Layernorm with Binary VGG net and BOP

### Training
```
TF_CPP_MIN_LOG_LEVEL=2 python classify.py train binaryvgg \
        --dataset cifar10 \
        --hparams-set bop \
        --hparams norm_layer="BatchNormalization",opt_param="{'threshold':5e-7, 'gamma':1e-4}",epochs=100 \ 
        --logdir ./../zoo/
```
You can choose your network (eg. binaryvgg), dataset (eg. CIFAR10), optimizer (eg. BOP) and your choice of hyperparameters BatchNormalization/LayerNormalization, threshold $\tau$ and adaptivity rate $\gamma$, number of epochs. 

This will save the network inside zoo as `./zoo/cifar10/binaryvgg/Log_number`

### Testing
Give the proper logdir where the trained model `Log_number` is saved. 
```
TF_CPP_MIN_LOG_LEVEL=2 python classify.py test binaryvgg \
        --dataset cifar10 \
        --hparams-set bop \
        --logdir ./../zoo/cifar10/binaryvgg/Log_number
```

### Plot the ablation studies
In plot.py, make `DenoisingAE = 0` and `compareBNLN = 0`, specify filepaths (e.g. filepath = `./../zoo/cifar10/binaryvgg`)in the code properly and run:
```
python plot.py 
```

## Compare Batch Normalization with Layer Normalization
Run the same commands as given in training for epochs=250. Take the two experiment logs (BN and LN) in new folder called `./../zoo/cifar10_250epochs` and give the same as filepath (e.g. `./../zoo/cifar10_250epochs`) in plot.py, then make `DenoisingAE = 0` and `compareBNLN = 1` and run:
```
python plot.py 
```

## Get training PSNR and training loss of denoising AutoEncoder and Binary denoising AutoEncoder
### Training of AE+Adam
```
TF_CPP_MIN_LOG_LEVEL=2 python ae.py train autoencoder \
        --dataset cifar10 \
        --preprocess-fn AE \
        --hparams-set traditional \
        --hparams norm_layer="BatchNormalization",epochs=100 \
        --logdir ./../zoo/denoisingAE/
```
The logs are saved in `./../zoo/denoisingAE/cifar10/autoencoder/Log_number`
### Training of BAE+Adam
```
TF_CPP_MIN_LOG_LEVEL=2 python ae.py train binaryae \
        --dataset cifar10 \
        --preprocess-fn AE \
        --hparams-set bnn \
        --hparams norm_layer="BatchNormalization",epochs=100 \
        --logdir ./../zoo/denoisingAE/bnn/
```
The logs are saved in `./../zoo/denoisingAE/bnn/cifar10/binaryae/Log_number`
### Training of BAE+BOP
```
TF_CPP_MIN_LOG_LEVEL=2 python ae.py train binaryae \
        --dataset cifar10 \
        --preprocess-fn AE \
        --hparams-set bop \
        --hparams norm_layer="BatchNormalization",opt_param="{'threshold':1e-8, 'gamma':1e-5}",epochs=100 \
        --logdir ./../zoo/denoisingAE/bop/
```
The logs are saved in `./../zoo/denoisingAE/bop/cifar10/binaryae/Log_number`


### Testing BAE+BOP
With trained model saved in directory `cifar10/binaryae/Log_number`, for testing, run:
```
TF_CPP_MIN_LOG_LEVEL=2 python ae.py test binaryae \
        --dataset cifar10 \
        --preprocess-fn AE \
        --hparams-set bop \
        --hparams norm_layer="BatchNormalization",opt_param="{'threshold':1e-8, 'gamma':1e-5}",epochs=100 \
        --logdir ./../zoo/cifar10/denoisingAE/bop/cifar10/binaryae/Log_number

```
### Visualize $5$ samples of test-dataset of CIFAR10 and their reconstruction using BAE+BOP
For BAE+BOP run:
```
TF_CPP_MIN_LOG_LEVEL=2 python ae.py vispics binaryae \
        --dataset cifar10 \
        --preprocess-fn AE \
        --hparams-set bop \
        --hparams norm_layer="BatchNormalization",opt_param="{'threshold':1e-8, 'gamma':1e-5}",epochs=100 \
        --logdir ./../zoo/cifar10/denoisingAE/bop/cifar10/binaryae/Log_number

```

Testing other models(AE+Adam and BAE+Adam) can be done in similar way. For visualization, by default the above command plots 2 sample with best PSNR, 1 sample with medium PSNR and 2 sample with high PSNR for BAE+BOP. Please follow the commands, commented inside vispics in the file ae.py to just find those 5 sample-ids and plot them for the other two models.

### Training accuracy of AE, and DAEs
To get the training accuracy for AE, run:
```
python plot.py
```
keeping `DenoisingAE = 1`, and giving proper filepath (e.g. for BOP `./../zoo/cifar10/denoisingAE/bop/cifar10/binaryae`) for the trained models.

If you find the content useful please cite our work.
```@article{nayak2019comprehensive,
  title={A comprehensive study on binary optimizer and its applicability},
  author={Nayak, Nancy and Raj, Vishnu and Kalyani, Sheetal},
  year={2019}
}
```
