# CIFAR10 dataset

from zookeeper import registry
from .tasks import ImageClassification

@registry.register_preprocess("cifar10")
