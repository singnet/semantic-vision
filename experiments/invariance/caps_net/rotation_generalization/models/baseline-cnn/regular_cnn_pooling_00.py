"""
Author: Maxim Peterson, max@singularitynet.io
"""
import tensorflow as tf
import numpy as np
import os, time, sys
import utils

# Image size
sx = 28
sy = 28
sc = 1

BATCH_SIZE = 128
IMG_DIM    = (sy, sx, sc)
NCLASSES      = 8

ang_min = -45
ang_max = 46

n_epochs = 300
global_step = tf.Variable(0, trainable=False)
learning_rate  = tf.train.exponential_decay(learning_rate=0.0001,
                                          global_step= global_step,
                                          decay_steps= 10000,
                                          decay_rate= 0.9,
                                          staircase=True)
beta1 = 0.95

LOGDIR = './result_mnist_rot_excl/regular_cnn_pooling_00/tb_log_within_train_all/'
SAVEDIR = './result_mnist_rot_excl/regular_cnn_pooling_00/model'

ERASE_LINE = '\x1b[2K'

# leaky relu alpha
leakyrelu_alpha    = 0.1

def lrelu(x):
    return tf.nn.relu(x) - leakyrelu_alpha * tf.nn.relu(-x)

# Placeholders for input data and the targets
x_input = tf.placeholder(tf.float32, (None, *IMG_DIM), name='Input')
y_target = tf.placeholder(tf.float32, [None, NCLASSES], name='Target')
phase_train = tf.placeholder(tf.bool, name='phase_train')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def model(x, phase_train, keep_prob, reuse = True):
    with tf.variable_scope('Basenet', reuse=reuse):
        x = tf.identity(x, name="input")
        x = tf.layers.conv2d(x, 64, 5, 1, padding='same', activation=lrelu, name='conv1')
        x = tf.contrib.layers.batch_norm(x,
                                         center=True, scale=True,
                                         is_training=phase_train,
                                         scope='bn1')

        x = tf.layers.max_pooling2d(x, pool_size=2, strides = 1, padding='same', name='pooling1')

        x = tf.layers.conv2d(x, 128, 5, 1, padding='same', activation=lrelu, name='conv2')
        x = tf.contrib.layers.batch_norm(x,
                                         center=True, scale=True,
                                         is_training=phase_train,
                                         scope='bn2')

        x = tf.layers.max_pooling2d(x, pool_size=2, strides=1, padding='same', name='pooling1')

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024, activation=lrelu, name='fc')

        x = tf.nn.dropout(x, keep_prob=keep_prob)
        x = tf.layers.dense(x, NCLASSES, activation=None, name='fc_out')
        x = tf.identity(x, name="activation_none")
        x = tf.nn.softmax(x, name="activation_softmax")
        x = tf.identity(x, name="output")
        return x

def train():
    with tf.variable_scope(tf.get_variable_scope()):
        basenet_output = model(x_input, phase_train=phase_train,
                                 keep_prob=keep_prob, reuse = False)
        basenet_output_test = model(x_input, phase_train=phase_train,
                                      keep_prob=keep_prob, reuse=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = basenet_output, labels=y_target), name="error_cost")
    correct_predictions_train = tf.equal(tf.argmax(basenet_output, 1), tf.argmax(y_target, 1))
    correct_predictions_test = tf.equal(tf.argmax(basenet_output_test, 1), tf.argmax(y_target, 1))
    accuracy_mean_train = tf.reduce_mean(tf.cast(correct_predictions_train, "float"))
    accuracy_mean_test = tf.reduce_mean(tf.cast(correct_predictions_test, "float"))
    accuracy_sum_train = tf.reduce_sum(tf.cast(correct_predictions_train, "float"))
    accuracy_sum_test = tf.reduce_sum(tf.cast(correct_predictions_test, "float"))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Basenet')
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    loss_summary = tf.summary.scalar(name='Loss', tensor=loss)
    accuracy_train_summary = tf.summary.scalar(name='Train Accuracy', tensor=accuracy_mean_train)
    accuracy_test_summary = tf.summary.scalar(name='Test Accuracy', tensor=accuracy_mean_test)

    summary_op = tf.summary.merge([loss_summary, accuracy_train_summary, accuracy_test_summary])

    saver = tf.train.Saver()


    # Prepare Training Data
    (x_train, y_train), (x_test, y_test) = utils.load_mnist_excluded()

    nTrain = x_train.shape[0]
    nTrainBatches = int(nTrain / BATCH_SIZE)

    nTest = x_test.shape[0]
    nTestBatches = int(nTest / BATCH_SIZE)
    step = 0
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(logdir=LOGDIR, graph=sess.graph)
        maxAcc = -1
        for i in range(n_epochs):
            print("------------------Epoch {}/{}------------------".format(i, n_epochs))
            mean_train_acc = 0
            for b in range(nTrainBatches):
                batch_x, batch_y = utils.get_random_mnist_batch(x_train, y_train, BATCH_SIZE)
                # Check what images are fed to the network
                if i==0 and b == 0:
                    utils.save_img_batch(batch_x, './DEBUG_TRAIN_BATCH.png')
                sess.run(optimizer, feed_dict={x_input: batch_x,
                                               y_target: batch_y,
                                               phase_train:True,
                                               keep_prob: 0.5})
                loss_val, step, lr, acc_mean_batch, acc_sum_batch, summary = sess.run(
                    [loss, global_step, learning_rate, accuracy_mean_train, accuracy_sum_train, summary_op],
                          feed_dict={x_input: batch_x,
                                     y_target: batch_y,
                                     phase_train: False,
                                     keep_prob: 1.0 })
                mean_train_acc +=  acc_sum_batch
                writer.add_summary(summary, global_step=step)
                sys.stdout.write(ERASE_LINE)
                sys.stdout.write("\r\rEpoch: {}, Global step: {}, Learning rate: {}, Iteration: {}/{}\tmean batch accuracy: {}".format(i, step, lr, b, nTrainBatches, acc_mean_batch))
                sys.stdout.flush()
                time.sleep(0.005)
                step += 1

            print("\n------------------Evaluation after epoch {}/{}------------------".format(i, n_epochs))
            print("TRAIN ACCURACY: {}".format(mean_train_acc/float(nTrainBatches*BATCH_SIZE)))
            mean_test_acc = 0
            for b in range(nTestBatches):
                batch_x = x_test[b*BATCH_SIZE:(b+1)*BATCH_SIZE, :, :, :]
                batch_y = y_test[b*BATCH_SIZE:(b+1)*BATCH_SIZE, :]
                batch_x = utils.rotate_batch_for_test(batch_x, batch_y, [ang_min, ang_max])
                if i==0 and b == 0:
                    utils.save_img_batch(batch_x, './DEBUG_TEST_BATCH.png')
                acc_sum_test_batch = sess.run( accuracy_sum_test, feed_dict={
                    x_input: batch_x,
                    y_target: batch_y,
                    phase_train: False,
                    keep_prob: 1.0 })
                mean_test_acc += acc_sum_test_batch

            nleft = nTest - int(nTestBatches * BATCH_SIZE)
            if nleft > 0:
                batch_x = x_test[(nTest - nleft):, :, :, :]
                batch_x = utils.rotate_batch_for_test(batch_x, batch_y, [ang_min, ang_max])
                batch_y = y_test[(nTest - nleft):, :]
                acc_sum_test_batch = sess.run(accuracy_sum_test, feed_dict={
                    x_input: batch_x,
                    y_target: batch_y,
                    phase_train: False,
                    keep_prob: 1.0})
                mean_test_acc += acc_sum_test_batch

            mean_test_acc = mean_test_acc / float(nTest)
            print("TEST ACCURACY: {}".format(mean_test_acc))

            if( mean_test_acc > maxAcc ):
                maxAcc = mean_test_acc
                saver.save(sess, SAVEDIR + '/intermediate_model')

        saver.save(sess, SAVEDIR + '/final-model')

train()