# Minimal Parameter Network.

What's the best error I can achieve on CIFAR10 using as few parameters as possible?


Currently - First test. 171,498 parameters.

Validation accuracy > 76%.


```
>>> model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_26 (InputLayer)        (None, 32, 32, 3)         0         
_________________________________________________________________
conv2d_139 (Conv2D)          (None, 32, 32, 32)        896       
_________________________________________________________________
batch_normalization_140 (Bat (None, 32, 32, 32)        128       
_________________________________________________________________
p_re_lu_140 (PReLU)          (None, 32, 32, 32)        32768     
_________________________________________________________________
dropout_128 (Dropout)        (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_140 (Conv2D)          (None, 32, 32, 32)        9248      
_________________________________________________________________
average_pooling2d_99 (Averag (None, 16, 16, 32)        0         
_________________________________________________________________
batch_normalization_141 (Bat (None, 16, 16, 32)        128       
_________________________________________________________________
p_re_lu_141 (PReLU)          (None, 16, 16, 32)        8192      
_________________________________________________________________
dropout_129 (Dropout)        (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_141 (Conv2D)          (None, 16, 16, 32)        9248      
_________________________________________________________________
average_pooling2d_100 (Avera (None, 8, 8, 32)          0         
_________________________________________________________________
batch_normalization_142 (Bat (None, 8, 8, 32)          128       
_________________________________________________________________
p_re_lu_142 (PReLU)          (None, 8, 8, 32)          2048      
_________________________________________________________________
dropout_130 (Dropout)        (None, 8, 8, 32)          0         
_________________________________________________________________
conv2d_142 (Conv2D)          (None, 8, 8, 32)          9248      
_________________________________________________________________
average_pooling2d_101 (Avera (None, 4, 4, 32)          0         
_________________________________________________________________
batch_normalization_143 (Bat (None, 4, 4, 32)          128       
_________________________________________________________________
p_re_lu_143 (PReLU)          (None, 4, 4, 32)          512       
_________________________________________________________________
dropout_131 (Dropout)        (None, 4, 4, 32)          0         
_________________________________________________________________
dense_27 (Dense)             (None, 4, 4, 512)         16896     
_________________________________________________________________
flatten_25 (Flatten)         (None, 8192)              0         
_________________________________________________________________
dense_28 (Dense)             (None, 10)                81930     
=================================================================
Total params: 171,498
Trainable params: 171,242
Non-trainable params: 256
_________________________________________________________________

```
