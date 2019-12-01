# Effect of latent weights on BNN with tensorflow
These codes are to find train and test accuracies (with binary and latent weights) for binary weight neural networks with different architectures and datasets on tensorflow.

## Run
For FullyCon with MNIST:
```
python mnist.py
``` 
For LENET5 with MNIST:
```
python mnistLENET5.py
```
For ConvNet with CIFAR10:
```
python cifar10.py

```
For LENET5 with CIFAR10:
```
python cifar10LENET5.py

```
## Accuracy

| DataSet:MNIST                     | FullyCon | LENET5 |
|-----------------------------------|----------|--------|
| Training accuracy                 |  100.00% | 99.99% |
| Test accuracy                     |  98.35%  | 99.20% |
| Test accuracy with latent weights |  99.01%  | 98.12% |
|-----------------------------------|----------|--------|
| DataSet:CIFAR10                   | ConvNet  | LENET5 |
|-----------------------------------|----------|--------|
| Training accuracy                 |  100.00% | 85.16% |
| Test accuracy                     |  81.72%  | 64.99% |
| Test accuracy with latent weights |  50.52%  | 54.62% |




## Ref
To verify the performance of BNN wih latent weights following the paper Latent Weights Do Not Exist(https://arxiv.org/pdf/1906.02107.pdf), we refered to the paper on Binarized Neural Networks:(https://arxiv.org/abs/1602.02830) and the code below.
[BinaryNet-on-tensorflow](https://github.com/uranusx86/BinaryNet-on-tensorflow)