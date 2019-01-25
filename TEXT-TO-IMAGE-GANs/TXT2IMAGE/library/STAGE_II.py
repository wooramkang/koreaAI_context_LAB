from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers import Concatenate, Dropout
from keras.layers import Activation
from keras.optimizers import Adam
from keras.optimizers import SGD
from TXT2IMAGE.library.utility.image_utils import combine_normalized_images, img_from_normalized_img
from keras import backend as K
from TXT2IMAGE.library.InstanceNormaliztion import InstanceNormalization

def build_STAGE_GEN(img_shape, gf):
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4, normalize=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = Activation('elu')(d)
        if normalize:
            d = BatchNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='elu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        #u = BatchNormalization()(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    channals = 3
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, gf * 2, normalize=False)
    d2 = conv2d(d1, gf * 4)
    d3 = conv2d(d2, gf * 8)
    d4 = conv2d(d3, gf * 16)
    #d5 = conv2d(d4, gf * 32)

    # Upsampling
    #u2 = deconv2d(d5, d4, gf * 32)
    u3 = deconv2d(d4, d3, gf * 16)
    u4 = deconv2d(u3, d2, gf * 8)
    u5 = deconv2d(u4, d1, gf * 4)
    u6 = deconv2d(u5, d0, gf * 2)

    '''
    #u = UpSampling2D(size=2)(u6)
    u = Conv2D(gf, kernel_size=4, strides=1, padding='same', activation='relu')(u6)
    u = InstanceNormalization()(u)
    u = Activation('elu')(u)
    '''

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(channals, kernel_size=5, strides=1,
                        padding='same')(u7)
    output_img = Activation('tanh')(output_img)

    model = Model(d0, output_img)
    optim = Adam(0.00008, 0.5)
    model.compile(loss='mae', optimizer=optim)
    model.summary()

    return model


def build_STAGE_DIS(img_shape, df):

    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = Activation('elu')(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=img_shape)

    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)
    d5 = d_layer(d4, df * 16)
    validity = Conv2D(1, kernel_size=3, strides=1, padding='same')(d5)

    model = Model(img, validity)
    optim = Adam(0.0002, 0.5)
    model.compile(loss='mse', optimizer=optim)
    model.summary()

    return model