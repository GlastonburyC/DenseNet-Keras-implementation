# DenseNet and DenseNet-BC implementation in Keras (Tensorflow Backend)

An implementation of the DenseNet architecture with growth rate = 12, L = 40 (~1M parameters).

Use:

~~~python
usage: densenet.py [-h] [--epochs EPOCHS] [--layers LAYERS] [--growth GROWTH]
                   [--filters FILTERS] [--classes CLASSES] [--blocks BLOCKS]
                   [--aug AUG] [--compression COMPRESSION]
                   [--upsample UPSAMPLE]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs to run
  --layers LAYERS       Number of layers defined as (depth-4)/no_dense_blocks
  --growth GROWTH       k - How many filters to add each convolution
  --filters FILTERS     Number of inital conv_filter
  --classes CLASSES     Number of output classes
  --blocks BLOCKS       Number of Dense blocks
  --aug AUG             To perform data augmentation, set to 1
  --compression COMPRESSION
                        [0-1] to specify bottleneck compression (0.5)
  --upsample UPSAMPLE   Upsample and concatenate initial conv to last conv
  ~~~


DenseNet-BC growth rate 12, inital conv filters 24, L = 100. (~800k parameters)

To implement Densenet with growth_rate = 12, Layers = 40 - CIFAR10 validation error rate 6.22%:

To reproduce:

~~~ 
python densenet.py --upsample 1
~~~

To implement Densenet-BC with Layers = 100, compression = 0.5 - Validation error rate 5.44%:

~~~ 
python densenet.py --upsample 1 --growth 12 --filters 24 --compression 0.5 --layers 16
~~~
