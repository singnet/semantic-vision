"""
License: Apache-2.0
Author: Suofei Zhang | Hang Yu
E-mail: zhangsuofei at njupt.edu.cn | hangyu5 at illinois.edu

Modified by Maxim Peterson, max@singularitynet.io
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg, get_coord_add, get_dataset_size_train, get_num_classes
import time
import numpy as np
import sys, os
import capsnet_em as net
import math
import utils
import logging
import daiquiri
from PIL import Image

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

IMAGE_SIZE = 28

sx = 28
sy = 28
sc = 1

IMG_DIM    = (sy, sx, sc)
Z_DIM      = 8

ERASE_LINE = '\x1b[2K'

def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :]
    return image

def get_input(batch_x_, batch_labels):
    img = tf.identity(batch_x_)
    lab = tf.identity(batch_labels)

    return [img, lab]

def main(args):
    """Get dataset hyperparameters."""
    assert len(args) == 2 and isinstance(args[1], str)
    dataset_name = args[1]
    logger.info('Using dataset: {}'.format(dataset_name))

    """Set reproduciable random seed"""
    tf.set_random_seed(1234)

    coord_add = get_coord_add(dataset_name)
    dataset_size = get_dataset_size_train(dataset_name)
    num_classes = get_num_classes(dataset_name)

    # Prepare Training Data
    (x_train, y_train), (x_test, y_test) = utils.load_mnist_excluded()

    with tf.Graph().as_default():#, tf.device('/cpu:0'):

        # Placeholders for input data and the targets
        x_input = tf.placeholder(tf.float32, (None, *IMG_DIM), name='Input')
        y_target = tf.placeholder(tf.int32, [None, ], name='Target')

        """Get global_step."""
        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        """Get batches per epoch."""
        num_batches_per_epoch = int(dataset_size / cfg.batch_size)

        """Use exponential decay leanring rate?"""
        lrn_rate = tf.maximum(tf.train.exponential_decay(
            1e-3, global_step, num_batches_per_epoch, 0.8), 1e-5)
        tf.summary.scalar('learning_rate', lrn_rate)
        opt = tf.train.AdamOptimizer()  # lrn_rate

        """Define the dataflow graph."""
        m_op = tf.placeholder(dtype=tf.float32, shape=())
        with tf.device('/gpu:0'):
            with slim.arg_scope([slim.variable]):#, device='/cpu:0'):
                sample_batch = tf.identity(x_input)
                batch_labels = tf.identity(y_target)
                batch_squash = tf.divide(sample_batch, 255.)
                batch_x = slim.batch_norm(sample_batch, center=False, is_training=True, trainable=True)
                output, pose_out = net.build_arch(batch_x, coord_add, is_train=True,
                                                  num_classes=num_classes)

                tf.logging.debug(pose_out.get_shape())
                loss, spread_loss, mse, reconstruction = net.spread_loss(
                    output, pose_out, batch_squash, batch_labels, m_op)
                sample_batch = tf.squeeze(sample_batch)
                decode_res_op = tf.concat( [sample_batch, 255*tf.reshape(reconstruction, [cfg.batch_size, IMAGE_SIZE, IMAGE_SIZE])], axis=0 )
                acc = net.test_accuracy(output, batch_labels)
                tf.summary.scalar('spread_loss', spread_loss)
                tf.summary.scalar('reconstruction_loss', mse)
                tf.summary.scalar('all_loss', loss)
                tf.summary.scalar('train__batch_acc', acc)

            """Compute gradient."""
            grad = opt.compute_gradients(loss)
            # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
            grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
                          for g, _ in grad if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]

        """Apply graident."""
        with tf.control_dependencies(grad_check):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = opt.apply_gradients(grad, global_step=global_step)

        """Set Session settings."""
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        """Set Saver."""
        var_to_save = [v for v in tf.global_variables(
        ) if 'Adam' not in v.name]  # Don't save redundant Adam beta/gamma
        saver = tf.train.Saver(var_list=var_to_save, max_to_keep=cfg.epoch)

        """Display parameters"""
        total_p = np.sum([np.prod(v.get_shape().as_list()) for v in var_to_save]).astype(np.int32)
        train_p = np.sum([np.prod(v.get_shape().as_list())
                          for v in tf.trainable_variables()]).astype(np.int32)
        logger.info('Total Parameters: {}'.format(total_p))
        logger.info('Trainable Parameters: {}'.format(train_p))

        # read snapshot
        # latest = os.path.join(cfg.logdir, 'model.ckpt-4680')
        # saver.restore(sess, latest)
        """Set summary op."""
        summary_op = tf.summary.merge_all()

        """Start coord & queue."""
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        """Set summary writer"""
        if not os.path.exists(cfg.logdir + '/caps/{}/train_log/'.format(dataset_name)):
            os.makedirs(cfg.logdir + '/caps/{}/train_log/'.format(dataset_name))
        summary_writer = tf.summary.FileWriter(
            cfg.logdir + '/caps/{}/train_log/'.format(dataset_name), graph=sess.graph)  # graph = sess.graph, huge!

        if not os.path.exists(cfg.logdir + '/caps/{}/images/'.format(dataset_name)):
            os.makedirs(cfg.logdir + '/caps/{}/images/'.format(dataset_name))

        """Main loop."""
        m_min = 0.2
        m_max = 0.9
        m = m_min
        max_iter = cfg.epoch * num_batches_per_epoch + 1

        for step in range(max_iter):
            tic = time.time()
            """"TF queue would pop batch until no file"""

            batch_x, batch_y = utils.get_random_mnist_batch(x_train, y_train, cfg.batch_size)

            try:
                _, loss_value, train_acc_val, summary_str, mse_value = sess.run(
                    [train_op, loss, acc, summary_op, mse], feed_dict={m_op: m,
                                                                  x_input: batch_x,
                                                                  y_target: batch_y})

                sys.stdout.write(ERASE_LINE)
                sys.stdout.write('\r\r%d/%d iteration finishes in ' % (step, max_iter) + '%f second' %
                            (time.time() - tic) + ' training accuracy = %f'%train_acc_val +
                                 ' loss=%f' % loss_value + '\treconstruction_loss=%f'%mse_value)
                sys.stdout.flush()
                time.sleep(0.001)

            except KeyboardInterrupt:
                sess.close()
                sys.exit()
            except tf.errors.InvalidArgumentError:
                logger.warning('%d iteration contains NaN gradients. Discard.' % step)
                continue
            else:
                """Write to summary."""
                if step % 10 == 0:
                    summary_writer.add_summary(summary_str, step)

                if step % 200 == 0:
                    images = sess.run(decode_res_op, feed_dict={m_op: m,
                                                                x_input: batch_x,
                                                                y_target: batch_y
                                                                })
                    image = combine_images(images)
                    img_name = cfg.logdir + '/caps/{}/images/'.format(dataset_name)+"/step_{}.png".format(str(step))
                    Image.fromarray(image.astype(np.uint8)).save(img_name)
                """Epoch wise linear annealling."""
                if (step % num_batches_per_epoch) == 0:
                    if step > 0:
                        m += (m_max - m_min) / (cfg.epoch * cfg.m_schedule)
                        if m > m_max:
                            m = m_max

                    """Save model periodically """
                    ckpt_path = os.path.join(
                    	cfg.logdir + '/caps/{}/'.format(dataset_name), 'model-{:.4f}.ckpt'.format(loss_value))
                    saver.save(sess, ckpt_path, global_step=step)

        ckpt_path = os.path.join(
            cfg.logdir + '/caps/{}/'.format(dataset_name), 'finall-model-{:.4f}.ckpt'.format(loss_value))
        saver.save(sess, ckpt_path, global_step=step)

        print('Training is finished!')

if __name__ == "__main__":
    tf.app.run()

