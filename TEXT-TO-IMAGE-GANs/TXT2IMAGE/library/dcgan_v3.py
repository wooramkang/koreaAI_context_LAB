from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, concatenate, LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Deconvolution2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dropout, Concatenate
from TXT2IMAGE.library.utility.image_utils import combine_normalized_images, img_from_normalized_img
from keras import backend as K
import numpy as np
from PIL import Image
import os
from keras.layers.merge import _Merge
from functools import partial
from TXT2IMAGE.library.utility.glove_loader import GloveModel
from TXT2IMAGE.library.InstanceNormaliztion import InstanceNormalization
from TXT2IMAGE.library.STAGE_II import build_STAGE_GEN
from TXT2IMAGE.library.SpectralNorm import ConvSN2D, DenseSN, ConvSN2DTranspose

#################
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((128, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


##############

class DCGanV3(object):
    model_name = 'dc-gan-v3'

    def __init__(self):
        K.set_image_dim_ordering('tf')
        self.generator = None
        self.discriminator = None
        self.model = None
        self.pos_dis_model = None
        self.neg_dis_model = None
        self.stageII = None
        self.neg_model = None
        self.img_width = 7
        self.img_height = 7
        self.img_channels = 3
        self.random_input_dim = 100
        self.text_input_dim = 300
        self.config = None
        self.glove_source_dir_path = './very_large_data'
        self.glove_model = GloveModel()

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, DCGanV3.model_name + '-config.npy')

    @staticmethod
    def get_weight_file_path(model_dir_path, model_type):
        return os.path.join(model_dir_path, DCGanV3.model_name + '-' + model_type + '-weights.h5')

    def clip_d_pos_weights(self):
        weights = [np.clip(w, -0.01, 0.01) for w in self.pos_dis_model.get_weights()]
        self.pos_dis_model.set_weights(weights)

    def clip_d_neg_weights(self):
        weights = [np.clip(w, -0.01, 0.01) for w in self.neg_dis_model.get_weights()]
        self.neg_dis_model.set_weights(weights)

    def build_discirm(self, text_input, img_input):

        text_layer = Dense(1024)(text_input)
        # text_layer = Dense(512)(text_input)

        # img_layer = MaxPooling2D(pool_size=(2, 2))(img_input)
        img_layer = Conv2D(64, kernel_size=(5, 5), strides=2, padding='same')(img_input)
        # img_layer = Conv2D(64, kernel_size=(5, 5), padding='same')(img_input)
        img_layer = BatchNormalization()(img_layer)
        # img_layer = InstanceNormalization()(img_layer)
        img_layer = Activation('elu')(img_layer)

        img_layer = Conv2D(128, kernel_size=(5, 5), strides=1, padding='same')(img_layer)
        # img_layer = Conv2D(64, kernel_size=(5, 5), padding='same')(img_input)
        img_layer = BatchNormalization()(img_layer)
        # img_layer = InstanceNormalization()(img_layer)
        img_layer = Activation('elu')(img_layer)

        # img_layer = MaxPooling2D(pool_size=(2, 2))(img_layer)
        img_layer = Conv2D(128, kernel_size=5, strides=2)(img_layer)
        # img_layer = Conv2D(128, kernel_size=5)(img_layer)
        # img_layer = InstanceNormalization()(img_layer)
        img_layer = BatchNormalization(momentum=0.5)(img_layer)
        img_layer = Activation('elu')(img_layer)

        img_layer = Conv2D(256, kernel_size=(5, 5), strides=1, padding='same')(img_layer)
        # img_layer = Conv2D(64, kernel_size=(5, 5), padding='same')(img_input)
        img_layer = BatchNormalization()(img_layer)
        # img_layer = InstanceNormalization()(img_layer)
        img_layer = Activation('elu')(img_layer)

        # img_layer = MaxPooling2D(pool_size=(2, 2))(img_layer)
        img_layer = Conv2D(256, kernel_size=5, strides=2)(img_layer)
        # img_layer = Conv2D(256, kernel_size=5)(img_layer)
        img_layer = BatchNormalization(momentum=0.5)(img_layer)
        img_layer = Activation('elu')(img_layer)

        img_layer = Conv2D(512, kernel_size=(5, 5), strides=1, padding='same')(img_layer)
        # img_layer = Conv2D(64, kernel_size=(5, 5), padding='same')(img_input)
        img_layer = BatchNormalization()(img_layer)
        # img_layer = InstanceNormalization()(img_layer)
        img_layer = Activation('elu')(img_layer)

        # img_layer = MaxPooling2D(pool_size=(2, 2))(img_layer)
        img_layer = Conv2D(1024, kernel_size=5, strides=2)(img_layer)
        # img_layer = Conv2D(512, kernel_size=5)(img_layer)
        img_layer = BatchNormalization(momentum=0.5)(img_layer)
        img_layer = Activation('elu')(img_layer)

        # img_layer = MaxPooling2D(pool_size=(2, 2))(img_layer)
        img_layer = Flatten()(img_layer)
        img_layer = Dense(1024)(img_layer)

        merged = concatenate([img_layer, text_layer])

        discriminator_layer = Activation('elu')(merged)
        discriminator_layer = Dense(1)(discriminator_layer)
        discriminator_output = Activation('sigmoid')(discriminator_layer)

        return discriminator_output

    def build_discirm_withSN(self, text_input, img_input):

        text_layer = Dense(1024)(text_input)

        # img_layer = MaxPooling2D(pool_size=(2, 2))(img_input)
        # img_layer = Conv2D(64, kernel_size=(5, 5), strides=2, padding='same')(img_input)
        img_layer = ConvSN2D(64, kernel_size=5, strides=2, kernel_initializer='glorot_uniform', padding='same')(
            img_input)
        img_layer = Activation('elu')(img_layer)

        img_layer = ConvSN2D(128, kernel_size=5, strides=1, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = Activation('elu')(img_layer)

        # img_layer = MaxPooling2D(pool_size=(2, 2))(img_layer)
        # img_layer = Conv2D(128, kernel_size=5, strides=2)(img_layer)
        img_layer = ConvSN2D(128, kernel_size=5, strides=2, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = Activation('elu')(img_layer)

        img_layer = ConvSN2D(256, kernel_size=5, strides=1, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = Activation('elu')(img_layer)

        # img_layer = MaxPooling2D(pool_size=(2, 2))(img_layer)
        # img_layer = Conv2D(256, kernel_size=5, strides=2)(img_layer)
        img_layer = ConvSN2D(256, kernel_size=5, strides=2, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = Activation('elu')(img_layer)

        img_layer = ConvSN2D(512, kernel_size=5, strides=1, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = Activation('elu')(img_layer)

        # img_layer = MaxPooling2D(pool_size=(2, 2))(img_layer)
        # img_layer = Conv2D(1024, kernel_size=5, strides=2)(img_layer)
        # img_layer = Conv2D(1024, kernel_size=5)(img_layer)
        img_layer = ConvSN2D(512, kernel_size=5, strides=2, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = Activation('elu')(img_layer)

        img_layer = ConvSN2D(1024, kernel_size=5, strides=1, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = Activation('elu')(img_layer)

        # img_layer = MaxPooling2D(pool_size=(2, 2))(img_layer)
        img_layer = Flatten()(img_layer)

        # img_layer = Dense(1024)(img_layer)
        img_layer = DenseSN(1024, kernel_initializer='glorot_uniform')(img_layer)

        merged = concatenate([img_layer, text_layer])
        discriminator_layer = Activation('tanh')(merged)

        # discriminator_layer = Dense(1)(discriminator_layer)
        discriminator_layer = DenseSN(1, kernel_initializer='glorot_uniform')(discriminator_layer)
        discriminator_output = Activation('sigmoid')(discriminator_layer)

        return discriminator_output

    def build_discirm_withSN_without_sigmoid(self, text_input, img_input):

        img_layer = ConvSN2D(64, kernel_size=5, strides=2, kernel_initializer='glorot_uniform', padding='same')(
            img_input)
        img_layer = LeakyReLU(0.2)(img_layer)
        img_layer = ConvSN2D(128, kernel_size=5, strides=1, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = LeakyReLU(0.2)(img_layer)
        img_layer = ConvSN2D(128, kernel_size=5, strides=2, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = LeakyReLU(0.2)(img_layer)
        img_layer = ConvSN2D(256, kernel_size=5, strides=1, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = LeakyReLU(0.2)(img_layer)
        img_layer = ConvSN2D(256, kernel_size=5, strides=2, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = LeakyReLU(0.2)(img_layer)

        img_layer = ConvSN2D(512, kernel_size=5, strides=1, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = LeakyReLU(0.2)(img_layer)
        img_layer = ConvSN2D(512, kernel_size=5, strides=2, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = LeakyReLU(0.2)(img_layer)

        img_layer = ConvSN2D(1024, kernel_size=5, strides=1, kernel_initializer='glorot_uniform', padding='same')(
            img_layer)
        img_layer = LeakyReLU(0.2)(img_layer)
        img_layer = Flatten()(img_layer)

        img_layer = DenseSN(1024, kernel_initializer='glorot_uniform')(img_layer)
        img_layer = LeakyReLU(0.2)(img_layer)

        text_layer = DenseSN(1024, kernel_initializer='glorot_uniform')(text_input)
        text_layer = LeakyReLU(0.2)(text_layer)
        merged = concatenate([img_layer, text_layer])

        discriminator_output = DenseSN(1, kernel_initializer='glorot_uniform', activation='linear')(merged)

        return discriminator_output

    def build_text_to_img_deconv(self, random_input, text_input1):

        # init_img_width = self.img_width // 4
        # init_img_height = self.img_height // 4

        init_img_width = 4
        init_img_height = 4

        ratio = self.random_input_dim / self.text_input_dim
        dense_dim = int(ratio * 1024)

        # random_dense = Dense(1024)(random_input)
        random_dense = Dense(dense_dim)(random_input)
        text_layer1 = Dense(1024)(text_input1)

        merged = concatenate([random_dense, text_layer1])
        generator_layer = Activation('relu')(merged)

        generator_layer = Dense(1024 * init_img_width * init_img_height)(generator_layer)
        generator_layer = Activation('relu')(generator_layer)

        generator_layer = Reshape((init_img_width, init_img_height, 1024),
                                  input_shape=(1024 * init_img_width * init_img_height,))(generator_layer)
        generator_layer = Deconvolution2D(512, kernel_size=5, strides=2, padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.8)(generator_layer)
        generator_layer = Activation('relu')(generator_layer)

        # generator_layer = Conv2D(256, kernel_size=5, padding='same')(generator_layer)
        # generator_layer = Activation('relu')(generator_layer)
        # generator_layer = BatchNormalization(momentum=0.8)(generator_layer)

        generator_layer = Deconvolution2D(256, kernel_size=5, strides=2, padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.8)(generator_layer)
        generator_layer = Activation('relu')(generator_layer)

        # generator_layer = Conv2D(128, kernel_size=5, padding='same')(generator_layer)
        # generator_layer = Activation('relu')(generator_layer)
        # generator_layer = BatchNormalization(momentum=0.8)(generator_layer)

        generator_layer = Deconvolution2D(128, kernel_size=5, strides=2, padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.8)(generator_layer)
        generator_layer = Activation('relu')(generator_layer)

        # generator_layer = Conv2D(64, kernel_size=5, padding='same')(generator_layer)
        # generator_layer = Activation('relu')(generator_layer)
        # generator_layer = BatchNormalization(momentum=0.8)(generator_layer)

        generator_layer = Deconvolution2D(64, kernel_size=5, strides=2, padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.8)(generator_layer)
        generator_layer = Activation('relu')(generator_layer)

        generator_layer = Conv2D(self.img_channels, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)

        # generator = Model([random_input, text_input1], generator_output)
        return generator_output

    def build_text_to_img(self, random_input, text_input1):

        # init_img_width = self.img_width // 4
        # init_img_height = self.img_height // 4

        init_img_width = 4
        init_img_height = 4

        ratio = self.random_input_dim / self.text_input_dim
        dense_dim = int(ratio * 1024)

        random_dense = Dense(dense_dim)(random_input)
        text_layer1 = Dense(1024)(text_input1)
        merged = concatenate([random_dense, text_layer1])
        generator_layer = LeakyReLU(0.2)(merged)

        generator_layer = Dense(1024 * init_img_width * init_img_height)(generator_layer)
        generator_layer = BatchNormalization(momentum=0.8)(generator_layer)
        generator_layer = LeakyReLU(0.2)(generator_layer)
        generator_layer = Reshape((init_img_width, init_img_height, 1024),
                                  input_shape=(1024 * init_img_width * init_img_height,))(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(512, kernel_size=5, padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.8)(generator_layer)
        # generator_layer = Activation('relu')(generator_layer)
        generator_layer = LeakyReLU(0.2)(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(256, kernel_size=5, padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.8)(generator_layer)
        # generator_layer = Activation('relu')(generator_layer)
        generator_layer = LeakyReLU(0.2)(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(128, kernel_size=5, padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.8)(generator_layer)
        # generator_layer = Activation('relu')(generator_layer)
        generator_layer = LeakyReLU(0.2)(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(64, kernel_size=5, padding='same')(generator_layer)
        generator_layer = BatchNormalization(momentum=0.8)(generator_layer)
        # generator_layer = Activation('relu')(generator_layer)
        generator_layer = LeakyReLU(0.2)(generator_layer)

        generator_layer = Conv2D(3, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)

        return generator_output

    def create_model(self):
        LEARNING_RATE = 0.00008
        random_input = Input(shape=(self.random_input_dim,))
        text_input = Input(shape=(self.text_input_dim,))
        img_input = Input(shape=(self.img_width, self.img_height, self.img_channels))

        # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = Adam(LEARNING_RATE, 0.9)
        # g_optim = RMSprop(lr=0.00005)

        # self.generator.compile(loss='mean_squared_error', optimizer="SGD")
        generator_model_p = self.build_text_to_img(random_input, text_input)
        self.generator = Model([random_input, text_input], generator_model_p)
        # self.generator.compile(loss='mean_squared_error', optimizer=g_optim)
        self.generator.compile(loss='mae', optimizer=g_optim)
        # self.generator.compile(loss=wasserstein_loss, optimizer=g_optim)
        print('generator: ', self.generator.summary())

        ####################
        dn_optim = Adam(LEARNING_RATE, 0.9)
        # dn_optim = RMSprop(lr=0.00005)
        ####################

        neg_dis_model_p = self.build_discirm(text_input, img_input)
        self.neg_dis_model = Model([text_input, img_input], neg_dis_model_p)
        neg_model_output = self.neg_dis_model([text_input, self.generator.output])
        print('discriminator: ', self.neg_dis_model.summary())
        self.neg_dis_model.compile(loss='binary_crossentropy', optimizer=dn_optim)
        # self.neg_dis_model.compile(loss='mean_squared_error', optimizer=g_optim)
        # self.neg_dis_model.compile(loss=wasserstein_loss, optimizer=dn_optim)
        self.neg_dis_model.trainable = False

        dp_optim = Adam(LEARNING_RATE, 0.9)
        # dp_optim = RMSprop(lr=0.00005)

        pos_dis_model_p = self.build_discirm(text_input, img_input)
        self.pos_dis_model = Model([text_input, img_input], pos_dis_model_p)
        pos_model_output = self.pos_dis_model([text_input, self.generator.output])
        print('discriminator: ', self.pos_dis_model.summary())
        self.pos_dis_model.compile(loss='binary_crossentropy', optimizer=dp_optim)
        # self.pos_dis_model.compile(loss='mean_squared_error', optimizer=g_optim)
        # self.pos_dis_model.compile(loss=wasserstein_loss, optimizer=dp_optim)
        self.pos_dis_model.trainable = False

        self.model = Model([random_input, text_input], [pos_model_output, neg_model_output])
        # optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        optim = Adam(LEARNING_RATE, 0.9)
        self.model.compile(loss='mean_squared_error', optimizer=g_optim)
        # optim = RMSprop(lr=0.00005)
        # self.model.compile(loss='binary_crossentropy', optimizer=optim)
        # self.model.compile(loss=wasserstein_loss, optimizer=optim)

        print('generator-discriminator: ', self.model.summary())

    def create_MONOmodel(self):
        LEARNING_RATE = 0.00008
        random_input = Input(shape=(self.random_input_dim,))
        text_input = Input(shape=(self.text_input_dim,))
        img_input = Input(shape=(self.img_width, self.img_height, self.img_channels))

        # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = Adam(LEARNING_RATE, 0.9)
        # g_optim = RMSprop(lr=0.00005)

        # self.generator.compile(loss='mean_squared_error', optimizer="SGD")
        generator_model_p = self.build_text_to_img(random_input, text_input)
        self.generator = Model([random_input, text_input], generator_model_p)
        # self.generator.compile(loss='mean_squared_error', optimizer=g_optim)
        self.generator.compile(loss='mae', optimizer=g_optim)
        # self.generator.compile(loss=wasserstein_loss, optimizer=g_optim)
        print('generator: ', self.generator.summary())

        dp_optim = Adam(LEARNING_RATE, 0.9)
        # dp_optim = RMSprop(lr=0.00005)

        pos_dis_model_p = self.build_discirm(text_input, img_input)
        self.pos_dis_model = Model([text_input, img_input], pos_dis_model_p)
        pos_model_output = self.pos_dis_model([text_input, self.generator.output])
        print('discriminator: ', self.pos_dis_model.summary())
        self.pos_dis_model.compile(loss='binary_crossentropy', optimizer=dp_optim)
        # self.pos_dis_model.compile(loss='mean_squared_error', optimizer=g_optim)
        # self.pos_dis_model.compile(loss=wasserstein_loss, optimizer=dp_optim)
        self.pos_dis_model.trainable = False

        self.model = Model([random_input, text_input], pos_model_output)
        # optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        optim = Adam(LEARNING_RATE, 0.9)
        self.model.compile(loss='mean_squared_error', optimizer=g_optim)
        # optim = RMSprop(lr=0.00005)
        # self.model.compile(loss='binary_crossentropy', optimizer=optim)
        # self.model.compile(loss=wasserstein_loss, optimizer=optim)

        print('generator-discriminator: ', self.model.summary())

    def create_SNmodel(self):
        LEARNING_RATE = 0.0002
        random_input = Input(shape=(self.random_input_dim,))
        text_input = Input(shape=(self.text_input_dim,))
        img_input = Input(shape=(self.img_width, self.img_height, self.img_channels))

        # self.generator.compile(loss='mean_squared_error', optimizer="SGD")
        generator_model_p = self.build_text_to_img(random_input, text_input)
        # generator_model_p = self.build_text_to_img_deconv(random_input, text_input)
        self.generator = Model([random_input, text_input], generator_model_p)
        # self.generator.compile(loss='mean_squared_error', optimizer=g_optim)
        # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = Adam(LEARNING_RATE, 0.7)
        # g_optim = RMSprop(lr=0.00005)
        self.generator.compile(loss='mse', optimizer=g_optim)
        # self.generator.compile(loss=wasserstein_loss, optimizer=g_optim)
        print('generator: ', self.generator.summary())

        d_optim = Adam(LEARNING_RATE, 0.7)
        pos_dis_model_p = self.build_discirm_withSN(text_input, img_input)
        self.pos_dis_model = Model([text_input, img_input], pos_dis_model_p)
        pos_model_output = self.pos_dis_model([text_input, self.generator.output])
        print('discriminator: ', self.pos_dis_model.summary())
        self.pos_dis_model.compile(loss='binary_crossentropy', optimizer=d_optim)
        # self.pos_dis_model.compile(loss='mean_squared_error', optimizer=g_optim)
        # self.pos_dis_model.compile(loss=wasserstein_loss, optimizer=dp_optim)
        self.pos_dis_model.trainable = False

        self.model = Model([random_input, text_input], pos_model_output)
        # optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        optim = Adam(LEARNING_RATE, 0.7)
        # optim = RMSprop(lr=0.00005)
        self.model.compile(loss='binary_crossentropy', optimizer=optim)
        # self.model.compile(loss=wasserstein_loss, optimizer=optim)

        print('generator-discriminator: ', self.model.summary())

    def create_WGANmodel(self):

        LEARNING_RATE = 0.00005
        random_input = Input(shape=(self.random_input_dim,))
        text_input = Input(shape=(self.text_input_dim,))
        img_input = Input(shape=(self.img_width, self.img_height, self.img_channels))

        generator_model_p = self.build_text_to_img(random_input, text_input)
        self.generator = Model([random_input, text_input], generator_model_p)
        # g_optim = RMSprop(lr=LEARNING_RATE)
        # self.generator.compile(loss=wasserstein_loss, optimizer=g_optim)
        print('generator: ', self.generator.summary())

        dp_optim = RMSprop(lr=LEARNING_RATE)
        pos_dis_model_p = self.build_discirm_withSN_without_sigmoid(text_input, img_input)
        self.pos_dis_model = Model([text_input, img_input], pos_dis_model_p)
        pos_model_output = self.pos_dis_model([text_input, self.generator.output])
        self.pos_dis_model.compile(loss=wasserstein_loss, optimizer=dp_optim)
        print('pos_discriminator: ', self.pos_dis_model.summary())
        self.pos_dis_model.trainable = False

        self.model = Model([random_input, text_input], pos_model_output)
        optim = RMSprop(lr=LEARNING_RATE)

        self.model.compile(loss=wasserstein_loss, optimizer=optim)

        print('generator-discriminator: ', self.model.summary())

    def create_model___trial(self):
        '''
        #the first model

        init_img_width = self.img_width // 4
        init_img_height = self.img_height // 4

        random_input = Input(shape=(self.random_input_dim,))
        text_input1 = Input(shape=(self.text_input_dim,))
        random_dense = Dense(self.random_input_dim)(random_input)
        text_layer1 = Dense(1024)(text_input1)

        merged = concatenate([random_dense, text_layer1])
        generator_layer = Activation('tanh')(merged)

        generator_layer = Dense(192 * init_img_width * init_img_height)(generator_layer)
        #generator_layer = BatchNormalization()(generator_layer)
        generator_layer = InstanceNormalization()(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)

        generator_layer = Reshape((init_img_width, init_img_height, 192),
                                  input_shape=(192 * init_img_width * init_img_height,))(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(96, kernel_size=5, padding='same')(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(self.img_channels, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)

        self.generator = Model([random_input, text_input1], generator_output)
        self.generator.compile(loss='mean_squared_error', optimizer="SGD")
        print('generator: ', self.generator.summary())

        '''
        init_img_width = self.img_width // 8
        init_img_height = self.img_height // 8

        random_input = Input(shape=(self.random_input_dim,))
        text_input1 = Input(shape=(self.text_input_dim,))
        random_dense = Dense(self.random_input_dim)(random_input)
        text_layer1 = Dense(1024)(text_input1)

        merged = concatenate([random_dense, text_layer1])

        generator_layer = Activation('tanh')(merged)

        generator_layer = Dense(512 * init_img_width * init_img_height)(generator_layer)
        # generator_layer = BatchNormalization()(generator_layer)
        # generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Activation('elu')(generator_layer)
        generator_layer = InstanceNormalization()(generator_layer)

        generator_layer = Reshape((init_img_width, init_img_height, 512),
                                  input_shape=(512 * init_img_width * init_img_height,))(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        # generator_layer = Deconvolution2D(384, kernel_size=5, strides=2, padding='same')(generator_layer)
        # generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Dropout(0.4)(generator_layer)
        # generator_layer = Activation('elu')(generator_layer)
        # generator_layer = InstanceNormalization()(generator_layer)
        # generator_layer = Concatenate()([generator_layer1, generator_layer2])
        '''
        #generator_layer = InstanceNormalization()(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        #generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Activation('elu')(generator_layer)
        #generator_layer = Dropout(0.4)(generator_layer)
        '''
        generator_layer = Conv2D(256, kernel_size=5, strides=1, padding='same')(generator_layer)
        # generator_layer = BatchNormalization()(generator_layer)
        # generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Dropout(0.4)(generator_layer)
        generator_layer = Activation('elu')(generator_layer)
        # generator_layer = InstanceNormalization()(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        # generator_layer = Deconvolution2D(256, kernel_size=5, strides=2, padding='same')(generator_layer)
        # generator_layer = Activation('elu')(generator_layer)
        # generator_layer = Concatenate()([generator_layer1, generator_layer2])
        '''
        #generator_layer = InstanceNormalization()(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        #generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Activation('elu')(generator_layer)
        #generator_layer = Dropout(0.4)(generator_layer)
        '''
        generator_layer = Conv2D(128, kernel_size=5, strides=1, padding='same')(generator_layer)
        # generator_layer = BatchNormalization()(generator_layer)
        # generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Dropout(0.4)(generator_layer)
        generator_layer = Activation('elu')(generator_layer)
        # generator_layer = InstanceNormalization()(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        # generator_layer = Deconvolution2D(144, kernel_size=5, strides=2, padding='same')(generator_layer)
        # generator_layer = Activation('elu')(generator_layer)
        # generator_layer = Concatenate()([generator_layer1, generator_layer2])
        '''
        #generator_layer = InstanceNormalization()(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        #generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Activation('elu')(generator_layer)
        #generator_layer = Dropout(0.4)(generator_layer)
        '''
        generator_layer = Conv2D(64, kernel_size=5, strides=1, padding='same')(generator_layer)
        # generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Dropout(0.4)(generator_layer)
        # generator_layer = Activation('relu')(generator_layer)
        generator_layer = Activation('elu')(generator_layer)
        generator_layer = InstanceNormalization()(generator_layer)

        generator_layer = Conv2D(self.img_channels, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)

        self.generator = Model([random_input, text_input1], generator_output)
        self.generator.compile(loss='mean_squared_error', optimizer="SGD")
        print('generator: ', self.generator.summary())

        text_input2 = Input(shape=(self.text_input_dim,))

        text_layer2 = Dense(1024)(text_input2)
        text_layer2 = Dropout(0.3)(text_layer2)

        img_input2 = Input(shape=(self.img_width, self.img_height, self.img_channels))

        img_layer2 = Conv2D(64, kernel_size=(5, 5), padding='same')(img_input2)
        img_layer2 = BatchNormalization()(img_layer2)
        img_layer2 = Activation('elu')(img_layer2)
        img_layer2 = Dropout(0.3)(img_layer2)

        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Conv2D(128, kernel_size=5)(img_layer2)
        img_layer2 = Activation('elu')(img_layer2)
        img_layer2 = Dropout(0.3)(img_layer2)

        '''
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Conv2D(1024, kernel_size=5)(img_layer2)
        img_layer2 = Activation('elu')(img_layer2)
        img_layer2 = Dropout(0.2)(img_layer2)
        '''

        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)

        # img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Flatten()(img_layer2)
        img_layer2 = Dense(1024)(img_layer2)

        merged = concatenate([img_layer2, text_layer2])

        discriminator_layer = Activation('tanh')(merged)
        discriminator_layer = Dense(1)(discriminator_layer)
        discriminator_output = Activation('sigmoid')(discriminator_layer)

        self.discriminator = Model([img_input2, text_input2], discriminator_output)

        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        d_optim = Adam(0.00005, 0.5)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

        print('discriminator: ', self.discriminator.summary())

        model_output = self.discriminator([self.generator.output, text_input1])

        self.model = Model([random_input, text_input1], model_output)
        self.discriminator.trainable = False

        # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = Adam(0.00005, 0.5)

        self.model.compile(loss='binary_crossentropy', optimizer=g_optim)

        print('generator-discriminator: ', self.model.summary())

    def load_model(self, model_dir_path):
        config_file_path = DCGanV3.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.img_width = self.config['img_width']
        self.img_height = self.config['img_height']
        self.img_channels = self.config['img_channels']
        self.random_input_dim = self.config['random_input_dim']
        self.text_input_dim = self.config['text_input_dim']
        self.glove_source_dir_path = self.config['glove_source_dir_path']
        self.create_SNmodel()
        self.glove_model.load(self.glove_source_dir_path, embedding_dim=self.text_input_dim)
        self.generator.load_weights(DCGanV3.get_weight_file_path(model_dir_path, 'generator'))
        self.pos_dis_model.load_weights(DCGanV3.get_weight_file_path(model_dir_path, 'pos_discriminator'))

        try:
            self.neg_dis_model.load_weights(DCGanV3.get_weight_file_path(model_dir_path, 'neg_discriminator'))
        except:
            pass

    def fit_DOUBLE(self, model_dir_path, image_label_pairs, epochs=None, batch_size=None, snapshot_dir_path=None,
                   snapshot_interval=None):
        if epochs is None:
            epochs = 100

        if batch_size is None:
            batch_size = 128

        if snapshot_interval is None:
            snapshot_interval = 20

        self.config = dict()
        self.config['img_width'] = self.img_width
        self.config['img_height'] = self.img_height
        self.config['random_input_dim'] = self.random_input_dim
        self.config['text_input_dim'] = self.text_input_dim
        self.config['img_channels'] = self.img_channels
        self.config['glove_source_dir_path'] = self.glove_source_dir_path
        self.create_model()
        # self.load_model(model_dir_path)
        self.glove_model.load(data_dir_path=self.glove_source_dir_path, embedding_dim=self.text_input_dim)
        config_file_path = DCGanV3.get_config_file_path(model_dir_path)

        np.save(config_file_path, self.config)
        noise = np.zeros((batch_size, self.random_input_dim))
        text_batch = np.zeros((batch_size, self.text_input_dim))

        for epoch in range(epochs):

            print("Epoch is", epoch)
            batch_count = int(image_label_pairs.shape[0] / batch_size)
            print("Number of batches", batch_count)
            for batch_index in range(batch_count):
                # Step 1: train the discriminator
                texts = []
                image_label_pair_batch = image_label_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]

                image_batch = []
                for index in range(batch_size):
                    image_label_pair = image_label_pair_batch[index]
                    normalized_img = image_label_pair[0]
                    text = image_label_pair[1]
                    texts.append(text)
                    image_batch.append(normalized_img)
                    text_batch[index, :] = self.glove_model.encode_doc(text, self.text_input_dim)
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)

                image_batch = np.array(image_batch)
                # image_batch = np.transpose(image_batch, (0, 2, 3, 1))

                #############################
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                generated_images = self.generator.predict([noise, text_batch], verbose=0)
                ##########################
                self.pos_dis_model.trainable = True
                d_loss = self.pos_dis_model.train_on_batch([text_batch, image_batch],
                                                           np.array([1] * batch_size))

                print("Epoch %d batch %d pos_d_batpos_loss : %f" % (epoch, batch_index, d_loss))
                # self.clip_d_pos_weights()

                self.pos_dis_model.trainable = False
                g_loss_p = self.model.train_on_batch([noise, text_batch],
                                                     [np.array([1] * batch_size), np.array([0] * batch_size)])

                print("Epoch %d batch %d pos_g_batpos_loss : " % (epoch, batch_index))
                print(g_loss_p)

                #############################

                self.pos_dis_model.trainable = True
                d_loss = self.pos_dis_model.train_on_batch([text_batch, generated_images
                                                            ],
                                                           np.array([0] * batch_size))

                print("Epoch %d batch %d neg_d_batpos_loss : %f" % (epoch, batch_index, d_loss))
                # self.clip_d_pos_weights()

                self.pos_dis_model.trainable = False
                g_loss_p = self.model.train_on_batch([noise, text_batch],
                                                     [np.array([1] * batch_size), np.array([0] * batch_size)])

                print("Epoch %d batch %d neg_g_batpos_loss : " % (epoch, batch_index))
                print(g_loss_p)

                #############################
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                generated_images = self.generator.predict([noise, text_batch], verbose=0)
                #############################

                self.neg_dis_model.trainable = True
                d_loss = self.neg_dis_model.train_on_batch([text_batch, image_batch],
                                                           np.array([0] * batch_size))

                print("Epoch %d batch %d pos_d_batneg_loss : %f" % (epoch, batch_index, d_loss))
                # self.clip_d_neg_weights()

                self.neg_dis_model.trainable = False
                g_loss_n = self.model.train_on_batch([noise, text_batch],
                                                     [np.array([1] * batch_size), np.array([0] * batch_size)])

                print("Epoch %d batch %d pos_g_batneg_loss : " % (epoch, batch_index))
                print(g_loss_n)

                ##########################
                self.neg_dis_model.trainable = True
                d_loss = self.neg_dis_model.train_on_batch([text_batch, generated_images
                                                            ],
                                                           np.array([1] * batch_size))

                print("Epoch %d batch %d neg_d_batneg_loss : %f" % (epoch, batch_index, d_loss))
                # self.clip_d_neg_weights()
                self.neg_dis_model.trainable = False
                g_loss_n = self.model.train_on_batch([noise, text_batch],
                                                     [np.array([1] * batch_size), np.array([0] * batch_size)])

                print("Epoch %d batch %d neg_g_batneg_loss : " % (epoch, batch_index))
                print(g_loss_n)

                #############################
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                generated_images = self.generator.predict([noise, text_batch], verbose=0)
                #############################
                self.pos_dis_model.trainable = True

                d_loss = self.pos_dis_model.train_on_batch(
                    [np.concatenate((text_batch, text_batch)), np.concatenate((image_batch, generated_images))
                     ],
                    np.array([1] * batch_size + [0] * batch_size))

                print("Epoch %d batch %d con_d_batpos_loss : %f" % (epoch, batch_index, d_loss))
                self.pos_dis_model.trainable = False

                # self.clip_d_pos_weights()

                #############################

                self.neg_dis_model.trainable = True

                d_loss = self.neg_dis_model.train_on_batch(
                    [np.concatenate((text_batch, text_batch)), np.concatenate((image_batch, generated_images))
                     ],
                    np.array([0] * batch_size + [1] * batch_size))
                print("Epoch %d batch %d con_d_batneg_loss : %f" % (epoch, batch_index, d_loss))
                self.neg_dis_model.trainable = False

                g_loss = self.model.train_on_batch([noise, text_batch],
                                                   [np.array([1] * batch_size), np.array([0] * batch_size)])

                print("Epoch %d batch %d cOn_g_batcon_loss : " % (epoch, batch_index))
                print(g_loss)

                # self.clip_d_neg_weights()

                #############################
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                generated_images = self.generator.predict([noise, text_batch], verbose=0)
                #############################

                if (epoch * batch_size + batch_index) % snapshot_interval == 0 and snapshot_dir_path is not None:
                    self.save_snapshots(generated_images, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch % 20, batch_index=batch_index % 20)
                    self.save_snapshots(image_batch, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch % 20, batch_index=(900 + batch_index % 20))

                    fp = open(
                        snapshot_dir_path + DCGanV3.model_name + str(epoch % 20) + "-" + str(batch_index % 20) + ".txt",
                        "w")
                    for t in texts:
                        fp.write(t + '\n')
                    fp.close()
                    print(epoch % 20)
                    print(batch_index % 20)
                    print("++++snap++++++")

                if (epoch * batch_size + batch_index) % 10 == 9:
                    self.generator.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'generator'), True)
                    self.pos_dis_model.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'pos_discriminator'),
                                                    True)
                    self.neg_dis_model.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'neg_discriminator'),
                                                    True)

        self.generator.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'generator'), True)
        self.pos_dis_model.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'pos_discriminator'), True)
        self.neg_dis_model.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'neg_discriminator'), True)

    def fit_mono(self, model_dir_path, image_label_pairs, epochs=None, batch_size=None, snapshot_dir_path=None,
                 snapshot_interval=None):
        if epochs is None:
            epochs = 100

        if batch_size is None:
            batch_size = 128

        if snapshot_interval is None:
            snapshot_interval = 20

        self.config = dict()
        self.config['img_width'] = self.img_width
        self.config['img_height'] = self.img_height
        self.config['random_input_dim'] = self.random_input_dim
        self.config['text_input_dim'] = self.text_input_dim
        self.config['img_channels'] = self.img_channels
        self.config['glove_source_dir_path'] = self.glove_source_dir_path
        # self.create_MONOmodel()
        self.create_SNmodel()

        # try:
        self.load_model(model_dir_path)
        # except:
        # print("there is no model-data")

        self.glove_model.load(data_dir_path=self.glove_source_dir_path, embedding_dim=self.text_input_dim)
        config_file_path = DCGanV3.get_config_file_path(model_dir_path)

        np.save(config_file_path, self.config)
        noise = np.zeros((batch_size, self.random_input_dim))
        text_batch = np.zeros((batch_size, self.text_input_dim))

        for epoch in range(epochs):

            print("Epoch is", epoch)
            batch_count = int(image_label_pairs.shape[0] / batch_size)
            print("Number of batches", batch_count)
            for batch_index in range(batch_count):
                # Step 1: train the discriminator
                texts = []
                image_label_pair_batch = image_label_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]

                image_batch = []
                for index in range(batch_size):
                    image_label_pair = image_label_pair_batch[index]
                    normalized_img = image_label_pair[0]
                    text = image_label_pair[1]
                    texts.append(text)
                    image_batch.append(normalized_img)
                    text_batch[index, :] = self.glove_model.encode_doc(text, self.text_input_dim)
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)

                image_batch = np.array(image_batch)
                # image_batch = np.transpose(image_batch, (0, 2, 3, 1))
                '''
                #############################
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                generated_images = self.generator.predict([noise, text_batch], verbose=0)
                ##########################

                self.pos_dis_model.trainable = True
                d_loss = self.pos_dis_model.train_on_batch([text_batch, image_batch],
                                                           np.array([1] * batch_size ))

                print("Epoch %d batch %d pos_d_batpos_loss : %f" % (epoch, batch_index, d_loss))
                #self.clip_d_pos_weights()

                self.pos_dis_model.trainable = False
                g_loss_p = self.model.train_on_batch([noise, text_batch], np.array([1] * batch_size))

                print("Epoch %d batch %d pos_g_batpos_loss : " % (epoch, batch_index))
                print(g_loss_p)

                #############################

                self.pos_dis_model.trainable = True
                d_loss = self.pos_dis_model.train_on_batch([text_batch, generated_images
                                                            ],
                                                           np.array([0] * batch_size))

                print("Epoch %d batch %d neg_d_batpos_loss : %f" % (epoch, batch_index, d_loss))
                #self.clip_d_pos_weights()

                self.pos_dis_model.trainable = False
                g_loss_p = self.model.train_on_batch([noise, text_batch], np.array([1] * batch_size))

                print("Epoch %d batch %d neg_g_batpos_loss : " % (epoch, batch_index))
                print(g_loss_p)

                self.pos_dis_model.trainable = True    
                d_loss = self.pos_dis_model.train_on_batch([np.concatenate((text_batch, text_batch)), np.concatenate((image_batch, generated_images))
                                                            ],
                                                           np.array([1] * batch_size + [0] * batch_size))

                print("Epoch %d batch %d con_d_batpos_loss : %f" % (epoch, batch_index, d_loss))
                self.pos_dis_model.trainable = False
                '''

                #############################
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                generated_images = self.generator.predict([noise, text_batch], verbose=0)
                #############################
                self.pos_dis_model.trainable = True
                d_loss = self.pos_dis_model.train_on_batch([text_batch, image_batch],
                                                           np.array([1] * batch_size))

                print("Epoch %d batch %d pos_d_batpos_loss : %f" % (epoch, batch_index, d_loss))
                # self.clip_d_pos_weights()

                d_loss = self.pos_dis_model.train_on_batch([text_batch, generated_images
                                                            ],
                                                           np.array([0] * batch_size))

                print("Epoch %d batch %d neg_d_batpos_loss : %f" % (epoch, batch_index, d_loss))
                self.pos_dis_model.trainable = False

                # self.clip_d_pos_weights()

                #############################

                g_loss = self.model.train_on_batch([noise, text_batch], np.array([1] * batch_size))

                print("Epoch %d batch %d cOn_g_batcon_loss : " % (epoch, batch_index))
                print(g_loss)

                # self.clip_d_neg_weights()

                #############################
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                generated_images = self.generator.predict([noise, text_batch], verbose=0)
                #############################

                if (epoch * batch_size + batch_index) % snapshot_interval == 0 and snapshot_dir_path is not None:
                    self.save_snapshots(generated_images, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch % 20, batch_index=batch_index % 20)
                    self.save_snapshots(image_batch, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch % 20, batch_index=(900 + batch_index % 20))

                    fp = open(
                        snapshot_dir_path + DCGanV3.model_name + str(epoch % 20) + "-" + str(batch_index % 20) + ".txt",
                        "w")
                    for t in texts:
                        fp.write(t + '\n')
                    fp.close()
                    print(epoch % 20)
                    print(batch_index % 20)
                    print("++++snap++++++")

                if (epoch * batch_size + batch_index) % 10 == 9:
                    self.generator.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'generator'), True)
                    self.pos_dis_model.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'pos_discriminator'),
                                                    True)

        self.generator.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'generator'), True)
        self.pos_dis_model.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'pos_discriminator'), True)

    def fit_wasserstein(self, model_dir_path, image_label_pairs, epochs=None, batch_size=None, snapshot_dir_path=None,
                        snapshot_interval=None):
        if epochs is None:
            epochs = 100

        if batch_size is None:
            batch_size = 128

        if snapshot_interval is None:
            snapshot_interval = 20

        self.config = dict()
        self.config['img_width'] = self.img_width
        self.config['img_height'] = self.img_height
        self.config['random_input_dim'] = self.random_input_dim
        self.config['text_input_dim'] = self.text_input_dim
        self.config['img_channels'] = self.img_channels
        self.config['glove_source_dir_path'] = self.glove_source_dir_path
        # self.create_MONOmodel()
        self.create_WGANmodel()
        self.glove_model.load(data_dir_path=self.glove_source_dir_path, embedding_dim=self.text_input_dim)
        config_file_path = DCGanV3.get_config_file_path(model_dir_path)
        np.save(config_file_path, self.config)
        noise = np.zeros((batch_size, self.random_input_dim))

        text_batch = np.zeros((batch_size, self.text_input_dim))

        for epoch in range(epochs):

            print("Epoch is", epoch)
            batch_count = int(image_label_pairs.shape[0] / batch_size)
            print("Number of batches", batch_count)
            for batch_index in range(batch_count):
                # Step 1: train the discriminator
                texts = []
                image_label_pair_batch = image_label_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]

                image_batch = []
                for index in range(batch_size):
                    image_label_pair = image_label_pair_batch[index]
                    normalized_img = image_label_pair[0]
                    text = image_label_pair[1]
                    texts.append(text)
                    image_batch.append(normalized_img)
                    text_batch[index, :] = self.glove_model.encode_doc(text, self.text_input_dim)
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)

                image_batch = np.array(image_batch)

                #######################################################################################
                self.pos_dis_model.trainable = True
                if epoch < 10:
                    n_critic = 10
                else:
                    n_critic = 5

                for i in range(n_critic):
                    #############################
                    for index in range(batch_size):
                        noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                    generated_images = self.generator.predict([noise, text_batch], verbose=0)
                    #############################

                    d_loss = self.pos_dis_model.train_on_batch([text_batch, image_batch],
                                                               np.array([1] * batch_size))

                    dx_loss = self.pos_dis_model.train_on_batch([text_batch, generated_images
                                                                 ],
                                                                np.array([-1] * batch_size))
                    d_loss = (d_loss + dx_loss) / 2
                    print("Epoch %d batch %d d_loss : %f" % (epoch, batch_index, d_loss))
                    self.clip_d_pos_weights()

                self.pos_dis_model.trainable = False
                #######################################################################################

                g_loss = self.model.train_on_batch([noise, text_batch], np.array([1] * batch_size))
                print("Epoch %d batch %d g_loss : %f" % (epoch, batch_index, g_loss))

                if (epoch * batch_size + batch_index) % snapshot_interval == 0 and snapshot_dir_path is not None:
                    self.save_snapshots(generated_images, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch % 20, batch_index=batch_index % 20)
                    self.save_snapshots(image_batch, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch % 20, batch_index=(900 + batch_index % 20))
                    self.generator.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'generator'), True)
                    self.pos_dis_model.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'pos_discriminator'),
                                                    True)
                    fp = open(
                        snapshot_dir_path + DCGanV3.model_name + str(epoch % 20) + "-" + str(batch_index % 20) + ".txt",
                        "w")
                    for t in texts:
                        fp.write(t + '\n')
                    fp.close()
                    print(epoch % 20)
                    print(batch_index % 20)
                    print("++++snap++++++")

        self.generator.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'generator'), True)
        self.pos_dis_model.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'pos_discriminator'), True)

    def fit_with_stageII(self, model_dir_path, image_label_pairs, s_label_pairs, epochs=None, batch_size=None,
                         snapshot_dir_path=None,
                         snapshot_interval=None):

        if epochs is None:
            epochs = 100

        if batch_size is None:
            batch_size = 128

        if snapshot_interval is None:
            snapshot_interval = 20

        self.config = dict()
        self.config['img_width'] = self.img_width
        self.config['img_height'] = self.img_height
        self.config['random_input_dim'] = self.random_input_dim
        self.config['text_input_dim'] = self.text_input_dim
        self.config['img_channels'] = self.img_channels
        self.config['glove_source_dir_path'] = self.glove_source_dir_path
        self.create_model()
        self.stageII = build_STAGE_GEN((64, 64, 3), 32)

        # self.load_model(model_dir_path)
        self.glove_model.load(data_dir_path=self.glove_source_dir_path, embedding_dim=self.text_input_dim)
        config_file_path = DCGanV3.get_config_file_path(model_dir_path)

        np.save(config_file_path, self.config)
        noise = np.zeros((batch_size, self.random_input_dim))
        text_batch = np.zeros((batch_size, self.text_input_dim))

        for epoch in range(epochs):
            print("Epoch is", epoch)
            batch_count = int(image_label_pairs.shape[0] / batch_size)
            print("Number of batches", batch_count)
            for batch_index in range(batch_count):
                # Step 1: train the discriminator
                texts = []
                image_label_pair_batch = image_label_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]

                image_batch = []
                for index in range(batch_size):
                    image_label_pair = image_label_pair_batch[index]
                    normalized_img = image_label_pair[0]
                    text = image_label_pair[1]
                    texts.append(text)
                    image_batch.append(normalized_img)
                    text_batch[index, :] = self.glove_model.encode_doc(text, self.text_input_dim)
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                image_batch = np.array(image_batch)

                s_label_pair_batch = s_label_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]

                simage_batch = []
                for index in range(batch_size):
                    s_label_pair = s_label_pair_batch[index]
                    normalized_img = s_label_pair[0]
                    simage_batch.append(normalized_img)

                simage_batch = np.array(simage_batch)

                # image_batch = np.transpose(image_batch, (0, 2, 3, 1))
                generated_images = self.generator.predict([noise, text_batch], verbose=0)

                s_loss = self.stageII.train_on_batch(generated_images, simage_batch)
                print("Epoch %d batch %d s_loss : %f" % (epoch, batch_index, s_loss))

                generated_simages = self.stageII.predict(generated_images)

                if (epoch * batch_size + batch_index) % snapshot_interval == 0 and snapshot_dir_path is not None:
                    self.save_snapshots(generated_images, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch % 20, batch_index=batch_index % 20)
                    self.save_snapshots(image_batch, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch % 20, batch_index=(900 + batch_index % 20))
                    self.save_snapshots(generated_simages, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch % 20, batch_index=(100 + batch_index % 20))

                    fp = open(
                        snapshot_dir_path + DCGanV3.model_name + str(epoch % 20) + "-" + str(batch_index % 20) + ".txt",
                        "w")
                    for t in texts:
                        fp.write(t + '\n')
                    fp.close()
                    print(epoch % 20)
                    print(batch_index % 20)
                    print("++++snap++++++")
                """
                """
                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch([np.concatenate((image_batch, generated_images)),
                                                            np.concatenate((text_batch, text_batch))],
                                                           np.array([1] * batch_size + [0] * batch_size))

                print("Epoch %d batch %d d_loss : %f" % (epoch, batch_index, d_loss))

                # Step 2: train the generator
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                """
                """
                self.discriminator.trainable = False
                g_loss = self.model.train_on_batch([noise, text_batch], np.array([1] * batch_size))

                print("Epoch %d batch %d g_loss : %f" % (epoch, batch_index, g_loss))
                if (epoch * batch_size + batch_index) % 10 == 9:
                    self.generator.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'generator'), True)
                    self.discriminator.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'discriminator'), True)
                    self.stageII.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'stage'), True)

        self.generator.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'generator'), True)
        self.discriminator.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'discriminator'), True)
        self.stageII.save_weights(DCGanV3.get_weight_file_path(model_dir_path, 'stage'), True)

    def generate_image_from_text(self, text):
        noise = np.zeros(shape=(1, self.random_input_dim))
        encoded_text = np.zeros(shape=(1, self.text_input_dim))
        encoded_text[0, :] = self.glove_model.encode_doc(text)
        noise[0, :] = np.random.uniform(-1, 1, self.random_input_dim)
        generated_images = self.generator.predict([noise, encoded_text], verbose=0)
        generated_image = generated_images[0]
        generated_image = generated_image * 127.5 + 127.5
        return Image.fromarray(generated_image.astype(np.uint8))

    def save_snapshots(self, generated_images, snapshot_dir_path, epoch, batch_index):
        image = combine_normalized_images(generated_images)
        img_from_normalized_img(image).save(
            os.path.join(snapshot_dir_path, DCGanV3.model_name + '-' + str(epoch) + "-" + str(batch_index) + ".png"))
