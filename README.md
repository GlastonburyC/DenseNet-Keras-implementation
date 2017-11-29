# DenseNet-like architecture in Keras

An implementation of the DenseNet bottleneck architecture with growth rate = 12.

3 dense blocks each with 16 layers giving rise to a 2.5M parameter network. Trained on CIFAR10 without data augmentation results in:

`train accuracy x
validation accuracy y
`
With (2x) dilated convolutions, validation accuracy improves to:
