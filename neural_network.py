import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np


batch_size = 32
epochs = 200

class NeuralNetwork(object):
    def __init__(self, netname):
        if netname == 'resnet':
            self.model = self.resnet()

    def lr_schedule(self, epoch):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def resnet_block(self, inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            conv_first (bool): conv-bn-activation (True) or
                activation-bn-conv (False)
        # Returns
            x (tensor): tensor as input to the next layer
        """
        if conv_first:
            x = Conv2D(num_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(inputs)
            x = BatchNormalization()(x)
            if activation:
                x = Activation(activation)(x)
            return x
        x = BatchNormalization()(inputs)
        if activation:
            x = Activation('relu')(x)
        x = Conv2D(num_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x)
        return x

    def resnet(self, input_shape, depth, num_classes=10):
        """
        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)
        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        inputs = Input(shape=input_shape)
        num_filters = 16
        num_sub_blocks = int((depth - 2) / 6)

        x = self.resnet_block(inputs=inputs)
        # Instantiate convolutional base (stack of blocks).
        for i in range(3):
            for j in range(num_sub_blocks):
                strides = 1
                is_first_layer_but_not_first_block = j == 0 and i > 0
                if is_first_layer_but_not_first_block:
                    strides = 2
                y = self.resnet_block(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = self.resnet_block(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if is_first_layer_but_not_first_block:
                    x = self.resnet_block(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters = 2 * num_filters

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def build_model(self, x_train, y_train):
       self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.lr_schedule(0)),
                      metrics=['accuracy'])

       self.model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 shuffle=True)
