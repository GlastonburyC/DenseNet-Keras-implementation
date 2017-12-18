# DenseNet and DenseNet-BC implementation in Keras (Tensorflow Backend)

An implementation of the DenseNet architecture with growth rate = 12, L = 40 (~1M parameters).

Trained on CIFAR10 without data augmentation:

```validation error rate 6.22%```

DenseNet-BC growth rate 12, inital conv filters 24, L = 100. (~800k parameters)

```Validation error rate 5.44%```


To implement Densenet with growth_rate = 12, Layers = 40:

~~~python 
python densenet.py 300, 12, 12, 16, 10, False, 3
~~~

To implement Densenet-BC with Layers = 100, compression = 0.5:

~~~python 
python densenet.py 300, 16, 12, 24, 10, False, 3, 0.5
~~~


*All experiments without data augmentation and with set seed, 42.

I used the following learning schedule for 300 epochs ```(SGD - initial_lr = 0.1, momentum=0.90)```

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
