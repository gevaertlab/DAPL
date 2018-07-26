import tensorflow as tf

from layers import *


def encoder4_d(input):
    fc_1 = fc(input, 'fc_1', 6896)
    fc_2 = fc(fc_1, 'fc_2', 2378)
    fc_3 = fc(fc_2, 'fc_3', 800)
    return fc_3
def decoder4_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 2378)
    fc_dec2 = fc(fc_dec1, 'fc_dec2', 6896)
    fc_dec3 = fc(fc_dec2, 'fc_dec3',17176)
    return fc_dec3
def autoencoder4_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder4_d(input_image)
        reconstructed_image = decoder4_d(encoding)
    return input_image, reconstructed_image


# No nonlinear activation function
def encoder10_d(input):
    fc_1 = fc(input, 'fc_1', 4000)
    fc_2 = fc(fc_1, 'fc_2', 800)
    return fc_2
def decoder10_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 4000)
    fc_dec2 = fc(fc_dec1, 'fc_dec2',17176)
    return fc_dec2
def autoencoder10_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder10_d(input_image)
        reconstructed_image = decoder10_d(encoding)
    return input_image, reconstructed_image


def encoder11_d(input):
    fc_1 = fc(input, 'fc_1', 4900)
    fc_2 = fc(fc_1, 'fc_2', 1400)
    fc_3 = fc(fc_2, 'fc_3', 400)
    return fc_3
def decoder11_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 1400)
    fc_dec2 = fc(fc_dec1, 'fc_dec2', 4900)
    fc_dec3 = fc(fc_dec2, 'fc_dec3',17176)  
    return fc_dec3

def autoencoder11_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder11_d(input_image)
        reconstructed_image = decoder11_d(encoding)
    return input_image, reconstructed_image



def encoder12_d(input):
    fc_1 = fc(input, 'fc_1', 9000)
    fc_2 = fc(fc_1, 'fc_2', 4500)
    return fc_2
def decoder12_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 9000)
    fc_dec2 = fc(fc_dec1, 'fc_dec2',17176)
    return fc_dec2
def autoencoder12_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder12_d(input_image)
        reconstructed_image = decoder12_d(encoding)
    return input_image, reconstructed_image



def encoder13_d(input):
    fc_1 = fc(input, 'fc_1', 6000)
    fc_2 = fc(fc_1, 'fc_2', 2000)
    return fc_2
def decoder13_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 6000)
    fc_dec2 = fc(fc_dec1, 'fc_dec2',17176)
    return fc_dec2
def autoencoder13_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder13_d(input_image)
        reconstructed_image = decoder13_d(encoding)
    return input_image, reconstructed_image


def encoder14_d(input):
    fc_1 = fc(input, 'fc_1', 2000)
    fc_2 = fc(fc_1, 'fc_2', 250)
    return fc_2
def decoder14_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 2000)
    fc_dec2 = fc(fc_dec1, 'fc_dec2',17176)
    return fc_dec2
def autoencoder14_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder14_d(input_image)
        reconstructed_image = decoder14_d(encoding)
    return input_image, reconstructed_image


def encoder15_d(input):
    fc_1 = fc(input, 'fc_1', 1700)
    fc_2 = fc(fc_1, 'fc_2', 170)
    return fc_2
def decoder15_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 1700)
    fc_dec2 = fc(fc_dec1, 'fc_dec2',17176) 
    return fc_dec2
def autoencoder15_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder15_d(input_image)
        reconstructed_image = decoder15_d(encoding)
    return input_image, reconstructed_image



def encoder16_d(input):
    fc_1 = fc(input, 'fc_1', 4000)    
    return fc_1
def decoder16_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 17176)  
    return fc_dec1
def autoencoder16_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder16_d(input_image)
        reconstructed_image = decoder16_d(encoding)
    return input_image, reconstructed_image


def encoder17_d(input):
    fc_1 = fc(input, 'fc_1', 2000)    
    return fc_1
def decoder17_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 17176)  
    return fc_dec1
def autoencoder17_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder17_d(input_image)
        reconstructed_image = decoder17_d(encoding)
    return input_image, reconstructed_image


def encoder18_d(input):
    fc_1 = fc(input, 'fc_1', 1000)    
    return fc_1
def decoder18_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 17176)  
    return fc_dec1
def autoencoder18_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder18_d(input_image)
        reconstructed_image = decoder18_d(encoding)
    return input_image, reconstructed_image


def encoder19_d(input):
    fc_1 = fc(input, 'fc_1', 8000)    
    return fc_1
def decoder19_d(input):
    fc_dec1 = fc(input, 'fc_dec1', 17176)  
    return fc_dec1
def autoencoder19_d(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder19_d(input_image)
        reconstructed_image = decoder19_d(encoding)
    return input_image, reconstructed_image

