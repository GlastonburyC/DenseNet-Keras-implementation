# DenseNet and DenseNet-BC architecture implementation in Keras

An implementation of the DenseNet architecture with growth rate = 12, Layers = 40.

Trained on CIFAR10 without data augmentation:

```validation error rate 6.37%```

Using following opt parameters for 300 epochs ```(SGD - initial_lr = 0.1, momentum=0.90)```

~~~~python
def step_decay(epoch):
    initial_lrate = 0.1
    if epoch < 150: 
        lrate = 0.1
    if epoch == 150:
        lrate = initial_lrate / 10
    if epoch > 150 and epoch < 225:
        lrate = initial_lrate / 10 
    if epoch >= 225:
        lrate = initial_lrate / 100
    return float(lrate)
~~~~

DenseNet as above but with concatenated initial dilations:

```Validation error rate: 6.44%```


DenseNet-BC growth rate 12, inital conv filters 24, depth = 100. (~800k parameters)

```Validation error rate y```
