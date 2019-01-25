from __future__ import print_function, division
import scipy

#from keras.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, AveragePooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.applications import vgg16
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

class CycleGAN():
    def __init__(self):

        self.img_size = 32
        self.img_rows = self.img_size
        self.img_cols = self.img_size


        self.text_dim = 200

        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.dataset_name = 'apple2orange'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))
        self.img_generator = None
        self.img_discriminator = None

        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        self.gf = 32
        self.df = 64

        self.lambda_cycle = 10.0
        self.lambda_id = 0.1 * self.lambda_cycle

        optimizer = Adam(0.0002, 0.5)

        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)

        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        self.d_A.trainable = False
        self.d_B.trainable = False

        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])

        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)
        self.imgG_txtD = self.build_imgG_txtD()

    def build_generator(self):
        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = BatchNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)


    def build_imgG_txtD(self):
        """
        self.img_rows = 128
        self.img_cols = 128
        self.img_size = 128

        self.text_dim = 200

        :return:
        """
        init_img_width = self.img_rows
        init_img_height = self.img_cols
        dense_dim = self.img_size*4
        out_dim = 1024

        raw_text_input = Input(shape=(self.text_dim,))
        #d0_raw = Dense(out_dim)(raw_text_input)
        #generator_layer = Activation('tanh')(d0_raw)

        #generator_layer = Dense(self.channels * init_img_width * init_img_height)(generator_layer)
        generator_layer = Dense(dense_dim*2 * init_img_width * init_img_height)(raw_text_input)
        generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Activation('relu')(generator_layer)

        generator_layer = Reshape((init_img_width, init_img_height, dense_dim*2),
                                  input_shape=(dense_dim*2 * init_img_width * init_img_height,))(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(dense_dim*2, kernel_size=5, padding='same')(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Activation('relu')(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(dense_dim, kernel_size=5, padding='same')(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Activation('relu')(generator_layer)

        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(self.channels, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('sigmoid', name = "GEN")(generator_layer)

        self.img_generator = Model(raw_text_input, generator_output)
        self.img_generator.compile(loss='mean_squared_error', optimizer="SGD")

        print('generator: ', self.img_generator.summary())

        text_input2 = Input(shape=(self.text_dim,))
        text_layer2 = Dense(out_dim)(text_input2)

        img_input2 = Input(shape=(self.img_rows*8, self.img_cols*8, self.channels))
        print(generator_output.shape)
        temp_ception = vgg16.VGG16(include_top=False,  input_tensor=img_input2, input_shape=(dense_dim*2,
                                                                                             dense_dim * 2,
                                                                                             self.channels),
                                   pooling=None, classes=None, weights=None)
        img_layer3 = AveragePooling2D((4, 4))(temp_ception.output)
        img_layer3 = Flatten()(img_layer3)
        img_layer3 = Dense(out_dim)(img_layer3)

        '''
        img_layer2 = Conv2D(64, kernel_size=(5, 5), padding='same')(
            img_input2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Conv2D(128, kernel_size=5)(img_layer2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)

        img_layer2 = Flatten()(img_layer2)
        img_layer2 = Dense(1024)(img_layer2)
        '''

        merged = concatenate([img_layer3, text_layer2])

        discriminator_layer = Activation('relu')(merged)
        discriminator_layer = Dense(1)(discriminator_layer)
        discriminator_output = Activation('sigmoid', name="DIS")(discriminator_layer)

        self.img_discriminator = Model([img_input2, text_input2], discriminator_output)

        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.img_discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

        print('discriminator: ', self.img_discriminator.summary())

        model_output = self.img_discriminator([self.img_generator.output, raw_text_input])
        model = Model(raw_text_input, model_output)

        return model
        #return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = BatchNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=200, batch_size=1, sample_interval=200)
