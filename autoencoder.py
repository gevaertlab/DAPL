import tensorflow as tf

from layers import *

# Autoencoder structures need to be adpated according to data dimension.
# More or less layers could be used with different node size. 
# We used a three layer autoencoder with bottleneck layer size of 800 for the Pan-cancer RNA sequencing data.

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


