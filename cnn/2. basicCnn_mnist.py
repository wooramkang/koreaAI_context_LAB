# source by kim jun hwa
# team context in ai-lab korea

# basic cnn

import tensorflow as tf

# data load

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# reshape the x to the input image size(4D TENSOR)

x_image = tf.reshape(x, [-1, 28,28,1]) # (# of images, width, height, channel)
##############################################################################################

# define the weight and bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    # tf.truncated_normal : normal distribution -> random number raise
    
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#############################################################################################
    
# define the conv2d, max_pool
    
# conv2d(strides = [1,1,1,1] -> stride 1, padding = 'SAME' -> output size is same as input)
def conv2d(x, W):
    
    conv = tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
    
    return conv

# 2x2 max pooling applied
def max_pool(x):
    
    pool = tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME')
    
    return pool
# input size = 28 x 28
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)
# output size = 14 x 14
# input size = 14 x 14
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)
# output size = 7 x 7

# connect to the fully connected layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# now we change tensor to vector

# softmax function need the vector type of image series

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# apply the dropout

keep_prob = tf.placeholder("float32")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# connecet to softmax
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(100)
    
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, 
                                  feed_dict = { x:batch[0], y_: batch[1], keep_prob : 0.9})
        print("step %d, training accuracy %g " %(i, train_accuracy))

    sess.run(train_step, feed_dict = {x:batch[0], y_ : batch[1], keep_prob : 0.8})
    
    print("test accuracy %g"% sess.run(accuracy, feed_dict ={ x: mnist.test.images, y_ : mnist.test.labels, keep_prob : 1.0}))
    
    