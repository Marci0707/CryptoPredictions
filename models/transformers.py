from keras import Input, Sequential, Model
from keras.activations import selu, elu
from keras.initializers.initializers_v1 import HeNormal, LecunNormal
from keras.initializers.initializers_v2 import GlorotNormal
from keras.layers import Dense, Concatenate, Flatten, Dropout, Add, LayerNormalization
import tensorflow as tf
from keras.optimizers import Adam
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops.losses.losses_impl import softmax_cross_entropy
from tensorflow_addons.metrics import F1Score
import keras.backend as K


def create_stacked_encoder(x_train, n_classes):
    inp = Input(shape=(x_train.shape[1], x_train.shape[2]))


    embed = Dense(units=8, activation=tf.keras.layers.LeakyReLU(alpha=0.3), kernel_initializer=HeNormal(),
                  kernel_regularizer='l1_l2',name='embedding')(inp)

    enc = SinePositionEncoding()(embed)

    added = Add()([enc, embed])

    stack = TransformerEncoder(intermediate_dim=20, num_heads=2, activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                               kernel_initializer=HeNormal())(added)
    stack = TransformerEncoder(intermediate_dim=20, num_heads=2, activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                               kernel_initializer=HeNormal())(stack)
    stack = TransformerEncoder(intermediate_dim=20, num_heads=2, activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                               kernel_initializer=HeNormal())(stack)
    flatten = Flatten()(stack)
    dense = Dense(units=10, activation=tf.keras.layers.LeakyReLU(alpha=0.3), kernel_initializer=HeNormal())(flatten)
    output = Dense(units=n_classes, activation='softmax')(dense)
    return Model(inp, output)




def create_encoder_block(x_train, n_classes):
    inp = Input(shape=(x_train.shape[1], x_train.shape[2]))

    norm = LayerNormalization()(inp)
    embed = Dense(units=8, activation=tf.keras.layers.LeakyReLU(alpha=0.3), kernel_initializer=HeNormal(),
                  kernel_regularizer='l1_l2',name='embedding')(norm)

    enc = SinePositionEncoding()(embed)

    added = Add()([enc, embed])

    stack = TransformerEncoder(intermediate_dim=20, num_heads=3, dropout=0.1, activation=tf.keras.layers.LeakyReLU(alpha=0.3),
                               kernel_initializer=HeNormal())(added)

    flatten = Flatten()(stack)
    output = Dense(units=n_classes, activation='softmax')(flatten)
    return Model(inp, output)
