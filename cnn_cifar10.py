# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 01:37:30 2020

@author: namcho

VGG-Net Traning hyper parameters

The ConvNet training procedure generally follows Krizhevsky et al. (2012) (except for sampling
the input crops from multi-scale training images, as explained later). Namely, the training is carried
out by optimising the multinomial logistic regression objective using mini-batch gradient descent
(based on back-propagation (LeCun et al., 1989)) with momentum. The batch size was set to 256,
momentum to 0.9. The training was regularised by weight decay (the L2 penalty multiplier set to
5 · 10−4
) and dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5).
The learning rate was initially set to 10−2
, and then decreased by a factor of 10 when the validation
set accuracy stopped improving. In total, the learning rate was decreased 3 times, and the learning
was stopped after 370K iterations (74 epochs). We conjecture that in spite of the larger number of
parameters and the greater depth of our nets compared to (Krizhevsky et al., 2012), the nets required
less epochs to converge due to (a) implicit regularisation imposed by greater depth and smaller conv.
filter sizes; (b) pre-initialisation of certain layers.

The initialisation of the network weights is important, since bad initialisation can stall learning due
to the instability of gradient in deep nets. To circumvent this problem, we began with training
the configuration A (Table 1), shallow enough to be trained with random initialisation. Then, when
training deeper architectures, we initialised the first four convolutional layers and the last three fullyconnected layers with the layers of net A (the intermediate layers were initialised randomly). We did
not decrease the learning rate for the pre-initialised layers, allowing them to change during learning.
For random initialisation (where applicable), we sampled the weights from a normal distribution
with the zero mean and 10−2 variance. The biases were initialised with zero. It is worth noting that
after the paper submission we found that it is possible to initialise the weights without pre-training
by using the random initialisation procedure of Glorot & Bengio (2010).

To obtain the fixed-size 224×224 ConvNet input images, they were randomly cropped from rescaled
training images (one crop per image per SGD iteration). To further augment the training set, the
crops underwent random horizontal flipping and random RGB colour shift (Krizhevsky et al., 2012).
Training image rescaling is explained below.

"""

import numpy as np
import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime

# Model name that we're gonna generate
model_name = 'CNN_V12_Cifar10'
class_count = 10
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
print('train_images.shape = ', train_images.shape)
print('train_labes.shape = ', train_labels.shape)

train_labels_oh = np.zeros((train_labels.shape[0], class_count))
for i in range(len(train_labels)):
        train_labels_oh[i,train_labels[i]] = 1

test_labels_oh = np.zeros((test_labels.shape[0], class_count))
for i in range(len(test_labels)):
       test_labels_oh[i, test_labels[i]] = 1

label_definations = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                                                'frog', 'horse', 'ship', 'truck']

# Resimlerdeki pixel degerleri 0-255 arasinda bir deger almaktadir. Normalize edelim
train_images = train_images / 255.0
test_images = test_images / 255.0

image_input = layers.Input(shape = (train_images.shape[1:]), dtype=tf.float32)

X = layers.Conv2D(filters = 32, kernel_size = (3,3), strides=(1,1), padding='same',
                  kernel_regularizer=keras.regularizers.l2(0.2))(image_input)
X = layers.Activation('relu')(X)

X = layers.Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), padding='same',
                  kernel_regularizer=keras.regularizers.l2(0.2))(X)
X = layers.Activation('relu')(X)

# Girdimizdeki bilgileri 16'lik bir derinlige aktaralim
X = layers.Conv2D(filters = 32, kernel_size = (2,2), strides = (2,2), padding = 'valid', name = 'CONV2D_1',
                  kernel_regularizer = keras.regularizers.l2(0.1))(X)
X = layers.Activation('relu', name = 'RELU_1')(X)
#X = layers.Dropout(0.3)(X)
#print('1. Layer aktivasyon X.shape = ', X.shape)

X = layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'valid', name = 'CONV2D_2',
                  kernel_regularizer=keras.regularizers.l2(0.1))(X)
#X = layers.BatchNormalization(axis = -1, name = 'BN_2')(X)
X = layers.Activation('relu')(X)

X = layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'valid', name = 'CONV2D_3',
                  kernel_regularizer=keras.regularizers.l2(0.1))(X)
X = layers.Activation('relu')(X)

X = layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (3,3), padding = 'valid',
                  kernel_regularizer=keras.regularizers.l2(0.1))(X)
X = layers.Activation('relu')(X)

X = layers.Flatten()(X)

X = layers.Dense(units = 256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.05))(X)
X = layers.Dropout(0.3)(X)
outputs_cnn = layers.Dense(units = 10, activation = 'softmax')(X)

model = tf.keras.Model(inputs = image_input, outputs = outputs_cnn, name = 'cifar10')
model.summary()
keras.utils.plot_model(model, model_name + ".png", show_shapes = True)

# Egitilmis modeli yukleyelim
model = keras.models.load_model('./model_output/' + model_name)

opt = keras.optimizers.SGD(learning_rate = 1e-3, momentum=0.9)
# Compiling ayarlayalim
model.compile(optimizer = opt,
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ['accuracy'])

#Tensorboard ayarlari
log_dir = "logs\\fit\\" + '_EPOCH20_' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_images, train_labels_oh, epochs = 20, batch_size=32,
          validation_data=(test_images, test_labels_oh), verbose = 1,
          callbacks=[tensorboard_callback])

score = model.evaluate(x=test_images, y=test_labels_oh, verbose=2)

print('Evaluation score = ', score)

# Modeli kaydedelim
model.save('./model_output/' + model_name)

# Egitim sonrasi performans
preds = model.predict(test_images[:40])
#print('preds: \n', preds)
preds_max = np.argmax(preds, axis = 1)
print('preds_max: \n', preds_max)
print('test_labes:\n', test_labels[:40].reshape(1, 40))