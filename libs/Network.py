import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class Network():
    def __init__(self, config: dict):
        self.model = self.build_model(config)

    def load_weights(self, path: str):
        self.model.load_weights(filepath=path)

    def save_weights(self, path: str):
        self.model.save_weights(filepath=f'{path}.h5', save_format='h5')

    def build_model(self, config: dict):
        bs = config['board_size']
        ks = config['kernel_size']
        nf = config['n_filters']
        input_shape = (bs, bs, 1)

        # Input convolutional layer
        inputs = Input(shape=input_shape)
        x = layers.Conv2D(filters=nf, kernel_size=ks, padding='same',
                          input_shape=input_shape, kernel_regularizer=l2(config['weight_decay']))(inputs)
        x = layers.BatchNormalization(axis=1)(x)
        conv_outputs = layers.Activation('relu')(x)

        for i in range(config['n_middle_blocks']):
            conv_outputs = layers.Conv2D(filters=nf, kernel_size=ks, padding='same', input_shape=input_shape,
                                         kernel_regularizer=l2(config['weight_decay']))(conv_outputs)
            conv_outputs = layers.BatchNormalization(axis=1)(conv_outputs)
            conv_outputs = layers.Activation('relu')(conv_outputs)

        head_inputs = input_shape if config.get('head_inputs_fixed', False) else input_shape[1:]

        # Policy head
        x = layers.Conv2D(filters=nf, kernel_size=ks, input_shape=head_inputs,
                          kernel_regularizer=l2(config['weight_decay']))(conv_outputs)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation('relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(bs**2, kernel_regularizer=l2(config['weight_decay']))(x)
        policy_outputs = layers.Activation('softmax')(x)

        # Value head
        x = layers.Conv2D(filters=1, kernel_size=1, input_shape=head_inputs,
                          kernel_regularizer=l2(config['weight_decay']))(conv_outputs)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation('relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(config['value_head_dense_layer_size'], kernel_regularizer=l2(config['weight_decay']))(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(1, kernel_regularizer=l2(config['weight_decay']))(x)
        value_output = layers.Activation('tanh')(x)

        model = Model(inputs, [policy_outputs, value_output])
        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(config['learning_rate']))
        return model

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int, epochs: int):
        self.model.fit(x, y, batch_size, epochs)

    def predict(self, x: np.ndarray):
        if len(x.shape) == 3:
            single_sample = np.expand_dims(x, axis=0)
            return self.model.predict(single_sample)
        elif len(x.shape) == 4:
            return self.model.predict(x)
        
        raise ValueError('x should either be of len(x.shape) = 3 (one sample) or len(x.shape) = 4 (batch)')
