import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import re
import time
import numpy as np
from numpy import matlib
import tensorflow as tf
import sys
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 
print('Using TensorFlow version: ', tf.__version__, ', GPU:', availale_GPUs)
print('Using Keras version: ', tf.keras.__version__)
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import savemat
import sys

def main():

    #dataset_train, dataset_valid = generate_pipeline_training(root_folder, batch_size=1)
    dataset_valid = generate_pipeline_training(root_folder, batch_size=1)

    gan3d = GAN3D(model_name, nx, ny, nz)

    generator, discriminator, generator_loss, discriminator_loss = gan3d.architecture04()
    generator_optimizer, discriminator_optimizer = gan3d.optimizer(learning_rate)

    checkpoint_dir = f"{root_folder}models/checkpoints_{model_name}"

    # Define checkpoint prefix

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # Generate checkpoint object to track the generator and discriminator architectures and optimizers

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )
    
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    ckpt_file = f"{root_folder}/models/checkpoints_{model_name}/ckpt-20"
    checkpoint.restore(ckpt_file).expect_partial()

    n_samp =  4000
    x_target = np.zeros((n_samp, nx,  1, nz, 3), np.float32)
    y_target = np.zeros((n_samp, nx, ny, nz, 3), np.float32)
    y_predic = np.zeros((n_samp, nx, ny, nz, 3), np.float32)

    itr = iter(dataset_valid)

    for idx in range(n_samp):
        print(idx)
        x, y = next(itr)      

        x_target[idx] = x.numpy()
        y_target[idx] = y.numpy() 

        y_predic[idx] = generator(np.expand_dims(x_target[idx], axis=0), training=False)
    
    error_u = np.mean((y_target[:,:,:,:,0]-y_predic[:,:,:,:,0])**2, axis=(0,1,3))
    error_v = np.mean((y_target[:,:,:,:,1]-y_predic[:,:,:,:,1])**2, axis=(0,1,3))
    error_w = np.mean((y_target[:,:,:,:,2]-y_predic[:,:,:,:,2])**2, axis=(0,1,3))
    
    X, Y, Z = read_channel_mesh_bin(NX, NY, NZ, LX, LZ)
    #print(NX, NY, NZ)
    #print(X.shape, Y.shape, Z.shape)
    #print(Y)
    
    print('Target',np.mean(y_target), np.std(y_target))
    print('Predic',np.mean(y_predic), np.std(y_predic))

    np.save('y_target.npy', y_target)
    np.save('y_predic.npy', y_predic)

    errors = {'y': Y[1:ny], 'error_u': error_u[1:ny], 'error_v': error_v[1:ny], 'error_w': error_w[1:ny]}
    savemat('error_data.mat', errors)


    """
    plt.rc('font', size=15) 
    plt.semilogx(200*Y[1:ny],error_u[1:], label='MSE U')
    plt.semilogx(200*Y[1:ny],error_v[1:], label='MSE V')
    plt.semilogx(200*Y[1:ny],error_w[1:], label='MSE W')
    plt.xlim([1, 200])
    plt.ylim([0, 1])
    plt.xlabel('Wall Distance $y^+$')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    plt.grid(which = 'minor', linestyle = '--')
    plt.savefig('error.png')
    plt.clf()

    plt.rc('font', size=15) 
    plt.semilogx(200*Y[1:ny],error_u[1:], label='MSE U')
    plt.semilogx(200*Y[1:ny],error_v[1:], label='MSE V')
    plt.semilogx(200*Y[1:ny],error_w[1:], label='MSE W')
    plt.semilogx([15, 30, 50, 100], [0.015, 0.065, 0.195, 0.53], label='mse u', linestyle=':', marker='D')
    plt.semilogx([15, 30, 50, 100], [0.02, 0.105, 0.30, 0.69], label='mse v', linestyle=':', marker='D')
    plt.semilogx([15, 30, 50, 100], [0.02, 0.085, 0.28, 0.69], label='mse w', linestyle=':', marker='D')
    plt.xlim([1, 200])
    plt.ylim([0, 1])
    plt.xlabel('Wall Distance $y^+$')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    plt.grid(which = 'minor', linestyle = '--')
    plt.savefig('error2D.png')
    plt.clf()


    rows=2
    cols=3
    ratio=0.5
    inches_per_pt = 1.0 / 72.27
    fig_width_pt = 400
    fig_width = fig_width_pt * inches_per_pt 
    fig_height = fig_width * rows / cols * ratio
    fig, axs = plt.subplots(rows,cols, figsize=(fig_width, fig_height), squeeze=False)
    axs[0,0].contourf(X,Z,y_target[0,:,22,:,0].T, vmin=-3, vmax=3, cmap='RdBu_r')
    axs[1,0].contourf(X,Z,y_predic[0,:,22,:,0].T, vmin=-3, vmax=3, cmap='RdBu_r')
    axs[0,1].contourf(X,Z,y_target[0,:,22,:,1].T, vmin=-3, vmax=3, cmap='PuOr_r')
    axs[1,1].contourf(X,Z,y_predic[0,:,22,:,1].T, vmin=-3, vmax=3, cmap='PuOr_r')
    axs[0,2].contourf(X,Z,y_target[0,:,22,:,2].T, vmin=-3, vmax=3, cmap='PiYG_r')
    axs[1,2].contourf(X,Z,y_predic[0,:,22,:,2].T, vmin=-3, vmax=3, cmap='PiYG_r')
    axs[0,0].set_xlim([0, np.pi])
    axs[0,0].set_ylim([0, np.pi/2])
    axs[0,1].set_xlim([0, np.pi])
    axs[0,1].set_ylim([0, np.pi/2])
    axs[0,2].set_xlim([0, np.pi])
    axs[0,2].set_ylim([0, np.pi/2])
    axs[1,0].set_xlim([0, np.pi])
    axs[1,0].set_ylim([0, np.pi/2])
    axs[1,1].set_xlim([0, np.pi])
    axs[1,1].set_ylim([0, np.pi/2])
    axs[1,2].set_xlim([0, np.pi])
    axs[1,2].set_ylim([0, np.pi/2])
    fig.savefig('prediction.png')
    # plt.clf()
    # plt.contourf(Z,X,x_target[0,:,0,:,1], vmin=-3, vmax=3, cmap='RdBu_r')
    # plt.axis('equal')
    # plt.savefig('input.png')
    """

    #print(Y*200)
    #print(Y[15]*200)
    
    return

class GAN3D(object):
  
    def __init__(self, model_name, nx, ny, nz, input_channels=3, output_channels=3, n_residual_blocks=32):
        
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.model_name = model_name
        self.n_residual_blocks = n_residual_blocks

        return


    def architecture01(self):
        
        """
            Generator model
        """

        def res_block_gen(model, kernal_size, filters, strides):
            
            gen = model
            model = layers.Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_last')(model) 
            model = layers.Add()([gen, model])
            
            return model

        def up_sampling_block(model, kernel_size, filters, strides):
            
            model = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            
            return model

        def up_sampling_block3(model, kernel_size, filters, strides):
            
            model = layers.UpSampling3D(size=(1, 3, 1), data_format='channels_last')(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            
            return model

        inputs = keras.Input(shape=(self.nx, 1, self.nz, self.input_channels), name='wall-input')

        conv_1 = layers.Conv3D(filters=64, kernel_size=9, strides=1, activation='linear', data_format='channels_last', padding='same')(inputs)

        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(conv_1)

        prelu_1 = up_sampling_block3(prelu_1, 3, 64, 1)

        res_block = prelu_1

        for index in range(self.n_residual_blocks):

            res_block = res_block_gen(res_block, 3, 64, 1)
        
        conv_2 = layers.Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(res_block)

        up_sampling = layers.Add()([prelu_1, conv_2])

        for index in range(int(4)):

            up_sampling = up_sampling_block(up_sampling, 3, 256, 1)

        outputs = layers.Conv3D(filters = self.output_channels, kernel_size = 9, strides = 1, padding = "same", data_format='channels_last')(up_sampling)

        # Connect input and output layers

        generator = keras.Model(inputs, outputs, name='GAN3D-Generator')

        """
            Discriminator model
        """

        def discriminator_block(model, filters, kernel_size, strides):
            
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)

            # Apply Leaky ReLU activation function

            model = layers.LeakyReLU(alpha = 0.2)(model)
            
            return model

        # Define input layer

        inputs = keras.Input(shape=(self.nx, self.ny, self.nz, self.output_channels), name='flow-input')

        # Apply a convolutional layer
        
        model = layers.Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(inputs)

        # Apply a Leaky ReLU activation function

        model = layers.LeakyReLU(alpha = 0.2)(model)

        # Apply 7 discriminator blocks 

        model = discriminator_block(model, 64, 3, 4)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        # Flatten the tensor into a vector

        model = layers.Flatten()(model)

        # Apply a fully-conncted layer 

        model = layers.Dense(1024)(model)

        # Apply a convolutional layer

        model = layers.LeakyReLU(alpha = 0.2)(model)

        # Apply a fully-conncted layer 

        model = layers.Dense(1)(model)

        # Apply a sigmoid connection function

        model = layers.Activation('sigmoid')(model) 

        # Connect input and output layers
        
        discriminator = keras.Model(inputs=inputs, outputs = model, name='GAN3D-Discriminator')

        """
            Generator loss
        """

        # Define generator loss as a function to be returned for its later use during the training

        def generator_loss(fake_y, y_predic, y_target):
            
            cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            
            adversarial_loss = tf.reshape(
                    cross_entropy(
                    np.ones(fake_y.shape) - np.random.random_sample(fake_y.shape) * 0.2, 
                    fake_y
                ),
                shape=(BATCH_SIZE_PER_REPLICA, 1, 1, 1)
            )

            mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            
            content_loss = mse(
                y_target, 
                y_predic
            )

            loss = content_loss + 1e-3*adversarial_loss

            scale_loss = tf.reduce_sum(loss, axis=(1,2,3)) / GLOBAL_BATCH_SIZE / self.nx / self.ny / self.nz

            return scale_loss 

        """
            Discriminator loss
        """

        # Define discriminator loss as a function to be returned for its later use during the training

        def discriminator_loss(real_y, fake_y):
            
            cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

            real_loss = cross_entropy(np.ones(real_y.shape) - np.random.random_sample(real_y.shape)*0.2, real_y)

            fake_loss = cross_entropy(np.random.random_sample(fake_y.shape)*0.2, fake_y)

            total_loss = 0.5 * (real_loss + fake_loss)
            
            return tf.nn.compute_average_loss(total_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        return generator, discriminator, generator_loss, discriminator_loss

    def architecture02(self):
        
        """
            Generator model
        """

        def res_block_gen(model, kernal_size, filters, strides):
            
            gen = model
            model = layers.Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_last')(model) 
            model = layers.Add()([gen, model])
            
            return model

        def up_sampling_block(model, kernel_size, filters, strides):
            
            model = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            
            return model

        def up_sampling_block3(model, kernel_size, filters, strides):
            
            model = layers.UpSampling3D(size=(1, 3, 1), data_format='channels_last')(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            
            return model

        inputs = keras.Input(shape=(self.nx, 1, self.nz, self.input_channels), name='wall-input')

        conv_1 = layers.Conv3D(filters=64, kernel_size=9, strides=1, activation='linear', data_format='channels_last', padding='same')(inputs)

        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(conv_1)

        prelu_1 = up_sampling_block3(prelu_1, 3, 64, 1)

        res_block = prelu_1

        for index in range(self.n_residual_blocks):

            res_block = res_block_gen(res_block, 3, 64, 1)
            
            if index == 7 or index == 14 or index == 21 or index == 28:

                res_block = up_sampling_block(res_block, 3, 64, 1)

                prelu_1 = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(prelu_1)
        
        conv_2 = layers.Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(res_block)

        conv_2 = layers.Add()([prelu_1, conv_2])

        outputs = layers.Conv3D(filters = self.output_channels, kernel_size = 9, strides = 1, padding = "same", data_format='channels_last')(conv_2)

        # Connect input and output layers

        generator = keras.Model(inputs, outputs, name='GAN3D-Generator')

        """
            Discriminator model
        """

        def discriminator_block(model, filters, kernel_size, strides):
            
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)

            # Apply Leaky ReLU activation function

            model = layers.LeakyReLU(alpha = 0.2)(model)
            
            return model

        # Define input layer

        inputs = keras.Input(shape=(self.nx, self.ny, self.nz, self.output_channels), name='flow-input')

        # Apply a convolutional layer
        
        model = layers.Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(inputs)

        # Apply a Leaky ReLU activation function

        model = layers.LeakyReLU(alpha = 0.2)(model)

        # Apply 7 discriminator blocks 

        model = discriminator_block(model, 64, 3, 4)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        # Flatten the tensor into a vector

        model = layers.Flatten()(model)

        # Apply a fully-conncted layer 

        model = layers.Dense(1024)(model)

        # Apply a convolutional layer

        model = layers.LeakyReLU(alpha = 0.2)(model)

        # Apply a fully-conncted layer 

        model = layers.Dense(1)(model)

        # Apply a sigmoid connection function

        model = layers.Activation('sigmoid')(model) 

        # Connect input and output layers
        
        discriminator = keras.Model(inputs=inputs, outputs = model, name='GAN3D-Discriminator')

        """
            Generator loss
        """

        # Define generator loss as a function to be returned for its later use during the training

        def generator_loss(fake_y, y_predic, y_target):
            
            cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            
            adversarial_loss = tf.reshape(
                    cross_entropy(
                    np.ones(fake_y.shape) - np.random.random_sample(fake_y.shape) * 0.2, 
                    fake_y
                ),
                shape=(BATCH_SIZE_PER_REPLICA, 1, 1, 1)
            )

            mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            
            content_loss = mse(
                y_target, 
                y_predic
            )

            loss = content_loss + 1e-3*adversarial_loss

            scale_loss = tf.reduce_sum(loss, axis=(1,2,3)) / GLOBAL_BATCH_SIZE / self.nx / self.ny / self.nz

            return scale_loss 

        """
            Discriminator loss
        """

        # Define discriminator loss as a function to be returned for its later use during the training

        def discriminator_loss(real_y, fake_y):
            
            cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

            real_loss = cross_entropy(np.ones(real_y.shape) - np.random.random_sample(real_y.shape)*0.2, real_y)

            fake_loss = cross_entropy(np.random.random_sample(fake_y.shape)*0.2, fake_y)

            total_loss = 0.5 * (real_loss + fake_loss)
            
            return tf.nn.compute_average_loss(total_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        return generator, discriminator, generator_loss, discriminator_loss

    def architecture03(self):
        
        """
            Generator model
        """

        def res_block_gen(model, kernal_size, filters, strides):
            
            gen = model
            model = layers.Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_last')(model) 
            model = layers.Add()([gen, model])
            
            return model

        def up_sampling_block(model, kernel_size, filters, strides):
            
            model = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            
            return model

        def up_sampling_block3(model, kernel_size, filters, strides):
            
            model = layers.UpSampling3D(size=(1, 3, 1), data_format='channels_last')(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            
            return model

        inputs = keras.Input(shape=(self.nx, 1, self.nz, self.input_channels), name='wall-input')

        conv_1 = layers.Conv3D(filters=64, kernel_size=9, strides=1, activation='linear', data_format='channels_last', padding='same')(inputs)

        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(conv_1)

        prelu_1 = up_sampling_block3(prelu_1, 3, 64, 1)

        res_block = prelu_1

        for index in range(self.n_residual_blocks):

            res_block = res_block_gen(res_block, 3, 64, 1)
            
            if index == 6 or index == 12 or index == 18 or index == 24:

                res_block = up_sampling_block(res_block, 3, 64, 1)

                prelu_1 = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(prelu_1)
        
        conv_2 = layers.Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(res_block)

        conv_2 = layers.Add()([prelu_1, conv_2])

        outputs = layers.Conv3D(filters = self.output_channels, kernel_size = 9, strides = 1, padding = "same", data_format='channels_last')(conv_2)

        # Connect input and output layers

        generator = keras.Model(inputs, outputs, name='GAN3D-Generator')

        """
            Discriminator model
        """

        def discriminator_block(model, filters, kernel_size, strides):
            
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)

            # Apply Leaky ReLU activation function

            model = layers.LeakyReLU(alpha = 0.2)(model)
            
            return model

        # Define input layer

        inputs = keras.Input(shape=(self.nx, self.ny, self.nz, self.output_channels), name='flow-input')

        # Apply a convolutional layer
        
        model = layers.Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(inputs)

        # Apply a Leaky ReLU activation function

        model = layers.LeakyReLU(alpha = 0.2)(model)

        # Apply 7 discriminator blocks 

        model = discriminator_block(model, 64, 3, 4)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        # Flatten the tensor into a vector

        model = layers.Flatten()(model)

        # Apply a fully-conncted layer 

        model = layers.Dense(1024)(model)

        # Apply a convolutional layer

        model = layers.LeakyReLU(alpha = 0.2)(model)

        # Apply a fully-conncted layer 

        model = layers.Dense(1)(model)

        # Apply a sigmoid connection function

        model = layers.Activation('sigmoid')(model) 

        # Connect input and output layers
        
        discriminator = keras.Model(inputs=inputs, outputs = model, name='GAN3D-Discriminator')

        """
            Generator loss
        """

        # Define generator loss as a function to be returned for its later use during the training

        def generator_loss(fake_y, y_predic, y_target):
            
            cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            
            adversarial_loss = tf.reshape(
                    cross_entropy(
                    np.ones(fake_y.shape) - np.random.random_sample(fake_y.shape) * 0.2, 
                    fake_y
                ),
                shape=(BATCH_SIZE_PER_REPLICA, 1, 1, 1)
            )

            mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            
            content_loss = mse(
                y_target, 
                y_predic
            )

            loss = content_loss + 1e-3*adversarial_loss

            scale_loss = tf.reduce_sum(loss, axis=(1,2,3)) / GLOBAL_BATCH_SIZE / self.nx / self.ny / self.nz

            return scale_loss 

        """
            Discriminator loss
        """

        # Define discriminator loss as a function to be returned for its later use during the training

        def discriminator_loss(real_y, fake_y):
            
            cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

            real_loss = cross_entropy(np.ones(real_y.shape) - np.random.random_sample(real_y.shape)*0.2, real_y)

            fake_loss = cross_entropy(np.random.random_sample(fake_y.shape)*0.2, fake_y)

            total_loss = 0.5 * (real_loss + fake_loss)
            
            return tf.nn.compute_average_loss(total_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        return generator, discriminator, generator_loss, discriminator_loss

    def architecture04(self):
        
        """
            Generator model
        """

        def res_block_gen(model, kernal_size, filters, strides):
            
            gen = model
            model = layers.Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same", data_format='channels_last')(model) 
            model = layers.Add()([gen, model])
            
            return model

        def up_sampling_block(model, kernel_size, filters, strides):
            
            model = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            
            return model

        def up_sampling_block3(model, kernel_size, filters, strides):
            
            model = layers.UpSampling3D(size=(1, 3, 1), data_format='channels_last')(model)
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)
            model = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(model)
            
            return model

        inputs = keras.Input(shape=(self.nx, 1, self.nz, self.input_channels), name='wall-input')

        conv_1 = layers.Conv3D(filters=64, kernel_size=9, strides=1, activation='linear', data_format='channels_last', padding='same')(inputs)

        prelu_1 = layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2,3])(conv_1)

        prelu_1 = up_sampling_block3(prelu_1, 3, 64, 1)

        res_block = prelu_1

        for index in range(self.n_residual_blocks):

            res_block = res_block_gen(res_block, 3, 64, 1)
            
            if index == 5 or index == 10 or index == 15 or index == 20:

                res_block = up_sampling_block(res_block, 3, 64, 1)

                prelu_1 = layers.UpSampling3D(size=(1, 2, 1), data_format='channels_last')(prelu_1)
        
        conv_2 = layers.Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(res_block)

        conv_2 = layers.Add()([prelu_1, conv_2])

        outputs = layers.Conv3D(filters = self.output_channels, kernel_size = 9, strides = 1, padding = "same", data_format='channels_last')(conv_2)

        # Connect input and output layers

        generator = keras.Model(inputs, outputs, name='GAN3D-Generator')

        """
            Discriminator model
        """

        def discriminator_block(model, filters, kernel_size, strides):
            
            model = layers.Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same", data_format='channels_last')(model)

            # Apply Leaky ReLU activation function

            model = layers.LeakyReLU(alpha = 0.2)(model)
            
            return model

        # Define input layer

        inputs = keras.Input(shape=(self.nx, self.ny, self.nz, self.output_channels), name='flow-input')

        # Apply a convolutional layer
        
        model = layers.Conv3D(filters = 64, kernel_size = 3, strides = 1, padding = "same", data_format='channels_last')(inputs)

        # Apply a Leaky ReLU activation function

        model = layers.LeakyReLU(alpha = 0.2)(model)

        # Apply 7 discriminator blocks 

        model = discriminator_block(model, 64, 3, 4)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        # Flatten the tensor into a vector

        model = layers.Flatten()(model)

        # Apply a fully-conncted layer 

        model = layers.Dense(1024)(model)

        # Apply a convolutional layer

        model = layers.LeakyReLU(alpha = 0.2)(model)

        # Apply a fully-conncted layer 

        model = layers.Dense(1)(model)

        # Apply a sigmoid connection function

        model = layers.Activation('sigmoid')(model) 

        # Connect input and output layers
        
        discriminator = keras.Model(inputs=inputs, outputs = model, name='GAN3D-Discriminator')

        """
            Generator loss
        """

        # Define generator loss as a function to be returned for its later use during the training

        def generator_loss(fake_y, y_predic, y_target):
            
            cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            
            adversarial_loss = tf.reshape(
                    cross_entropy(
                    np.ones(fake_y.shape) - np.random.random_sample(fake_y.shape) * 0.2, 
                    fake_y
                ),
                shape=(BATCH_SIZE_PER_REPLICA, 1, 1, 1)
            )

            mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
            
            content_loss = mse(
                y_target, 
                y_predic
            )

            loss = content_loss + 1e-3*adversarial_loss

            scale_loss = tf.reduce_sum(loss, axis=(1,2,3)) / GLOBAL_BATCH_SIZE / self.nx / self.ny / self.nz

            return scale_loss 

        """
            Discriminator loss
        """

        # Define discriminator loss as a function to be returned for its later use during the training

        def discriminator_loss(real_y, fake_y):
            
            cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

            real_loss = cross_entropy(np.ones(real_y.shape) - np.random.random_sample(real_y.shape)*0.2, real_y)

            fake_loss = cross_entropy(np.random.random_sample(fake_y.shape)*0.2, fake_y)

            total_loss = 0.5 * (real_loss + fake_loss)
            
            return tf.nn.compute_average_loss(total_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        return generator, discriminator, generator_loss, discriminator_loss



    def optimizer(self, learning_rate):

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

        return generator_optimizer, discriminator_optimizer


def generate_pipeline_training(root_folder, validation_split=1, shuffle_buffer=200, batch_size=4, n_prefetch=4):

    tfr_path = f"{root_folder}tfrecords/test/"
    tfr_files = sorted([os.path.join(tfr_path,f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path,f))])
    regex = re.compile(f'.tfrecords')
    tfr_files = ([string for string in tfr_files if re.search(regex, string)])
    
    n_samples_per_tfr = np.array([int(s.split('.')[-2][-3:]) for s in tfr_files])
    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]
    cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))
    tot_samples_per_ds = sum(n_samples_per_tfr)
    n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-3][-3:])

    tfr_files = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_tfr_loaded_per_ds]

    n_samp_train = int(sum(n_samples_per_tfr) * (1 - validation_split))
    n_samp_valid = sum(n_samples_per_tfr) - n_samp_train

    (n_files_train, samples_train_left) = np.divmod(n_samp_train, n_samples_per_tfr[0])

    if samples_train_left > 0:

        n_files_train += 1

    tfr_files_train = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_files_train]
    n_tfr_left = np.sum(np.where(cumulative_samples_per_tfr < samples_train_left, 1, 0)) + 1
    
    if sum([int(s.split('.')[-2][-2:]) for s in tfr_files_train]) != n_samp_train:

        shared_tfr = tfr_files_train[-1]
        tfr_files_valid = [shared_tfr]
    else:
        
        shared_tfr = ''
        tfr_files_valid = list()

    tfr_files_valid.extend([string for string in tfr_files if string not in tfr_files_train])
    tfr_files_valid = sorted(tfr_files_valid)

    shared_tfr_out = tf.constant(shared_tfr)
    n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)
    n_samples_loaded_per_tfr = list()

    if n_tfr_loaded_per_ds>1:

        n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds-1])
        n_samples_loaded_per_tfr.append(tot_samples_per_ds - cumulative_samples_per_tfr[n_tfr_loaded_per_ds-2])

    else:

        n_samples_loaded_per_tfr.append(tot_samples_per_ds)

    n_samples_loaded_per_tfr = np.array(n_samples_loaded_per_tfr)

    #tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)
    tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, seed=686)

    if n_tfr_left>1:

        samples_train_shared = samples_train_left - cumulative_samples_per_tfr[n_tfr_left-2]
        n_samples_tfr_shared = n_samples_loaded_per_tfr[n_tfr_left-1]

    else:

        samples_train_shared = samples_train_left
        n_samples_tfr_shared = n_samples_loaded_per_tfr[0]

    #tfr_files_train_ds = tfr_files_train_ds.interleave(
    #    lambda x : tf.data.TFRecordDataset(x).take(samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3],sep='-')[0], tf.int32)-1)), 
    #    cycle_length=16, 
    #    num_parallel_calls=tf.data.experimental.AUTOTUNE
    #)

    tfr_files_val_ds = tfr_files_val_ds.interleave(
        lambda x : tf.data.TFRecordDataset(x).skip(samples_train_shared).take(n_samples_tfr_shared - samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3],sep='-')[0], tf.int32)-1)),
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    #dataset_train = tfr_files_train_ds.map(lambda x: tf_parser(x, root_folder), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset_train = dataset_train.shuffle(shuffle_buffer)
    #dataset_train = dataset_train.batch(batch_size=batch_size)
    #dataset_train = dataset_train.prefetch(n_prefetch)

    dataset_valid = tfr_files_val_ds.map(lambda x: tf_parser(x, root_folder), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_valid = dataset_valid.shuffle(shuffle_buffer)
    dataset_valid = dataset_valid.batch(batch_size=batch_size)
    dataset_valid = dataset_valid.prefetch(n_prefetch)

    return dataset_valid


@tf.function
def tf_parser(rec, root_folder):

    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),     
        'nx': tf.io.FixedLenFeature([], tf.int64),    
        'ny': tf.io.FixedLenFeature([], tf.int64),   
        'nz': tf.io.FixedLenFeature([], tf.int64), 
        'x': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     
        'y': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),     
        'z': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),    
        'raw_u': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), 
        'raw_v': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True), 
        'raw_w': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_p': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_p': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_tx': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_tz': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_tx': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_tz': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }

    parsed_rec = tf.io.parse_single_example(rec, features)

    i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)

    nx = tf.cast(parsed_rec['nx'], tf.int32)
    ny = tf.cast(parsed_rec['ny'], tf.int32)
    nz = tf.cast(parsed_rec['nz'], tf.int32)

    filename = f"{root_folder}tfrecords/scaling.npz"

    # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    U_mean = np.expand_dims(np.load(filename)['U_mean'], axis=-1)[:,:64,:,:]
    V_mean = np.expand_dims(np.load(filename)['V_mean'], axis=-1)[:,:64,:,:]
    W_mean = np.expand_dims(np.load(filename)['W_mean'], axis=-1)[:,:64,:,:]
    PB_mean = np.expand_dims(np.load(filename)['PB_mean'], axis=-1)
    #PT_mean = np.expand_dims(np.load(filename)['PT_mean'], axis=-1)
    TBX_mean = np.expand_dims(np.load(filename)['TBX_mean'], axis=-1)
    TBZ_mean = np.expand_dims(np.load(filename)['TBZ_mean'], axis=-1)
    #TTX_mean = np.expand_dims(np.load(filename)['TTX_mean'], axis=-1)
    #TTZ_mean = np.expand_dims(np.load(filename)['TTZ_mean'], axis=-1)

    # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    U_std = np.expand_dims(np.load(filename)['U_std'], axis=-1)[:,:64,:,:]
    V_std = np.expand_dims(np.load(filename)['V_std'], axis=-1)[:,:64,:,:]
    W_std = np.expand_dims(np.load(filename)['W_std'], axis=-1)[:,:64,:,:]
    PB_std = np.expand_dims(np.load(filename)['PB_std'], axis=-1)
    #PT_std = np.expand_dims(np.load(filename)['PT_std'], axis=-1)
    TBX_std = np.expand_dims(np.load(filename)['TBX_std'], axis=-1)
    TBZ_std = np.expand_dims(np.load(filename)['TBZ_std'], axis=-1)
    #TTX_std = np.expand_dims(np.load(filename)['TTX_std'], axis=-1)
    #TTZ_std = np.expand_dims(np.load(filename)['TTZ_std'], axis=-1)

    # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

    flow = (tf.reshape(parsed_rec['raw_u'], (nx, ny, nz, 1)) - U_mean) / U_std
    flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_v'], (nx, ny, nz, 1)) - V_mean) / V_std), -1)
    flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_w'], (nx, ny, nz, 1)) - W_mean) / W_std), -1)
    
    flow = tf.where(tf.math.is_nan(flow), tf.zeros_like(flow), flow)

    wall = (tf.reshape(parsed_rec['raw_p'], (nx, 1, nz, 1)) - PB_mean) / PB_std
    wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_tx'], (nx, 1, nz, 1)) - TBX_mean) / TBX_std), -1)
    wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_tz'], (nx, 1, nz, 1)) - TBZ_mean) / TBZ_std), -1)
    

    return wall, flow[:,0:48,:,:]


def read_channel_mesh_bin(NX, NY, NZ, LX, LZ):

    X = np.arange(NX) * LX / NX
    Y = np.load('coordY.npy')
    Z = np.arange(NZ) * LZ / NZ
    
    return X, Y, Z


if __name__ == '__main__':

    nx = 64
    ny = 48
    nz = 64
    NX = 64
    NY = 128
    NZ = 64
    LX = np.pi
    LZ = np.pi / 2
    learning_rate = 1e-4
    model_name = 'architecture-B04'
    root_folder = '/    /'
    GLOBAL_BATCH_SIZE = 1
    BATCH_SIZE_PER_REPLICA = 4

    main()