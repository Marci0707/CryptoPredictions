from keras import Input, Sequential, Model
from keras.activations import selu, elu
from keras.initializers.initializers_v1 import HeNormal, LecunNormal
from keras.initializers.initializers_v2 import GlorotNormal
from keras.layers import Dense, Concatenate, Flatten
import tensorflow as tf
from keras.optimizers import Adam
from keras_nlp.layers import TransformerEncoder
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops.losses.losses_impl import softmax_cross_entropy
from tensorflow_addons.metrics import F1Score
import keras.backend as K


class Time2Vector(Layer):

    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)



    def call(self, x,*args,**kwargs):
        x = tf.math.reduce_mean(x[:,:,:], axis=-1) # Convert (batch, seq_len, 5) to (batch, seq_len)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1) # (batch, seq_len, 1)
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1) # (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1) # (batch, seq_len, 2


def hyperopt_2towers(hp):
    inp = Input(shape=(9,17), name='norm_inp')
    inp_transposed = Input(shape=(17,9), name='trans_inp')

    embedding_units_spac = hp.Int('embedding_units_spac',min_value=10,max_value=50,step=10)
    embedding_units_temp = hp.Int('embedding_units_temp',min_value=5,max_value=25,step=5)

    intermediate_units_spac = hp.Int('intermediate_units_spac',min_value=5,max_value=65,step=7)
    intermediate_units_temp = hp.Int('embedding_units_temp',min_value=5,max_value=65,step=7)

    initializer_map = {'relu': HeNormal(),'elu': HeNormal(),'selu': LecunNormal()}

    activation_function = hp.Choice('activation_function', values=['relu','elu','selu'])

    embedding_spac = Dense(units=embedding_units_spac, activation=activation_function, kernel_initializer=initializer_map[activation_function],kernel_regularizer='l1')(inp)
    embedding_temp = Dense(units=embedding_units_temp, activation=activation_function, kernel_initializer=initializer_map[activation_function],kernel_regularizer='l1')(inp_transposed)

    tower_height_spac = hp.Choice('tower_height_spac', values=[1,2,3,4])
    tower_height_temp = hp.Choice('tower_height_temp', values=[1,2,3,4])

    n_heads_spac = hp.Choice('n_heads_spac', values=[1,2,3])
    n_heads_temp = hp.Choice('n_heads_temp', values=[1,2,3])

    dropout_spac = hp.Choice('dropout_spac', values=[0.0,0.15,0.3])
    dropout_temp = hp.Choice('dropout_spac', values=[0.0,0.15,0.3])

    last_spac = embedding_spac
    for i in range(tower_height_spac):
        tower_spac = TransformerEncoder(intermediate_dim=intermediate_units_spac, num_heads=n_heads_spac
                                        , dropout=dropout_spac, activation=activation_function
                                        , kernel_initializer=initializer_map[activation_function])(last_spac)
        last_spac = tower_spac

    last_temp = embedding_temp
    for i in range(tower_height_temp):
        tower_temp = TransformerEncoder(intermediate_dim=intermediate_units_temp, num_heads=n_heads_temp
                                        , dropout=dropout_temp, activation=activation_function
                                        , kernel_initializer=initializer_map[activation_function])(last_temp)
        last_temp = tower_temp

    flatten_space = Flatten()(last_spac)
    flatten_temp = Flatten()(last_temp)

    concat = Concatenate(axis=1)([flatten_space, flatten_temp])

    last_layer_units = hp.Int('embedding_units_temp',min_value=5,max_value=50,step=7)

    dense = Dense(units=last_layer_units, activation=activation_function, kernel_initializer=initializer_map[activation_function],kernel_regularizer='l1')(concat)
    output = Dense(units=3, activation='softmax', kernel_initializer=GlorotNormal())(dense)

    model = Model([inp, inp_transposed], output)

    start_lr = hp.Choice('start_lr', values=[0.001,0.0005,0.0001])

    model.compile(loss=softmax_cross_entropy, optimizer=Adam(learning_rate=start_lr), metrics=['accuracy',F1Score(num_classes=3,average='macro')])

    return model



def create_2towers(x_train, x_train_trans, n_classes):
    inp = Input(shape=(x_train.shape[1], x_train.shape[2]), name='norm_inp')
    inp_transposed = Input(shape=(x_train_trans.shape[1], x_train_trans.shape[2]), name='trans_inp')

    embedding_temp = Dense(units=40, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer=HeNormal(),
                           kernel_regularizer='l1')(inp_transposed)
    embedding_spac = Dense(units=40, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer=HeNormal(),
                           kernel_regularizer='l1')(inp)

    tower_spac = TransformerEncoder(intermediate_dim=30, num_heads=1, dropout=0.1,
                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                    kernel_initializer=HeNormal())(embedding_spac)
    tower_spac = TransformerEncoder(intermediate_dim=30, num_heads=1, dropout=0.1,
                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                    kernel_initializer=HeNormal())(tower_spac)

    flatten_space = Flatten()(tower_spac)

    tower_temp = TransformerEncoder(intermediate_dim=30, num_heads=1, dropout=0.1,
                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                    kernel_initializer=HeNormal(),name='tower_1_')(embedding_temp)
    tower_temp = TransformerEncoder(intermediate_dim=30, num_heads=1, dropout=0.1,
                                    activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                    kernel_initializer=HeNormal())(tower_temp)
    flatten_temp = Flatten()(tower_temp)

    concat = Concatenate(axis=1)([flatten_space, flatten_temp])
    dense = Dense(units=30, activation='relu',kernel_initializer='HeNormal',kernel_regularizer='l1')(concat)
    output = Dense(units=n_classes, activation='softmax')(dense)

    return Model([inp, inp_transposed], output)


def create_stacked_encoder(x_train, n_classes):
    inp = Input(shape=(x_train.shape[1], x_train.shape[2]))

    embed = Dense(units=20, activation=tf.keras.layers.LeakyReLU(alpha=0.3), kernel_initializer=HeNormal(),
                  kernel_regularizer='l1')(inp)
    stack = TransformerEncoder(intermediate_dim=20, num_heads=2, dropout=0.1, activation='relu',
                               kernel_initializer=HeNormal())(embed)
    stack = TransformerEncoder(intermediate_dim=20, num_heads=1, dropout=0.1, activation='relu',
                               kernel_initializer=HeNormal())(stack)
    stack = TransformerEncoder(intermediate_dim=20, num_heads=1, dropout=0.1, activation='relu',
                               kernel_initializer=HeNormal())(stack)
    stack = TransformerEncoder(intermediate_dim=20, num_heads=1, dropout=0.1, activation='relu',
                               kernel_initializer=HeNormal())(stack)
    flatten = Flatten()(stack)
    output = Dense(units=n_classes)(flatten)
    return Model(inp, output)

def create_encoder_block(x_train, n_classes):


    inp = Input(shape=(x_train.shape[1], x_train.shape[2]))

    embed = Time2Vector(x_train.shape[-2])(inp)

    stack = TransformerEncoder(intermediate_dim=10, num_heads=2, dropout=0.1, activation='relu',
                               kernel_initializer=HeNormal(),name='encoder_0')(embed)
    stack = TransformerEncoder(intermediate_dim=10, num_heads=2, dropout=0.1, activation='relu',
                               kernel_initializer=HeNormal(),name='encoder_1')(stack)
    stack = TransformerEncoder(intermediate_dim=10, num_heads=2, dropout=0.1, activation='relu',
                               kernel_initializer=HeNormal(),name='encoder_2')(stack)
    flatten = Flatten()(stack)
    dense = Dense(units=40, activation='relu',kernel_initializer='HeNormal',kernel_regularizer='l1')(flatten)
    output = Dense(units=n_classes, activation='softmax')(dense)
    return Model([inp], output)