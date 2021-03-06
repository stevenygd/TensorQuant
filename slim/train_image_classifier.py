# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

from Quantize import Quantizers
from Quantize import QSGD
from Quantize import QRMSProp

import utils
import numpy as np

tf.app.flags.DEFINE_string(
    'train_dir', 'tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_epoches', 50, 'Number epoches.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'eval_every_n_steps', 1000,
    'The frequency with which evaluation is done.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 60,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')


#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'mnist', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'lenet', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

######################
# Quantization Flags #
######################

tf.app.flags.DEFINE_string(
    'extr_grad_quantizer', '', 'Word width and fractional digits of gradient quantizer.'
    'If None, no quantizer is applied. Passed as `w,q`.')

tf.app.flags.DEFINE_string(
    'intr_grad_quantizer', '', 'Word width and fractional digits of gradient quantizer.'
    'If None, no quantizer is applied. Passed as `w,q`.')

###############################
# Quantization
###############################
tf.app.flags.DEFINE_string(
    'intr_qmap', '', 'Location of intrinsic quantizer map.'
    'If empty, no quantizer is applied.')

tf.app.flags.DEFINE_string(
    'extr_qmap', '', 'Location of extrinsic quantizer map.'
    'If empty, no quantizer is applied.')

tf.app.flags.DEFINE_string(
    'weight_qmap', '', 'Location of weight quantizer map.'
    'If empty, no quantizer is applied.')



def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate, quantizer=None):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """

  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    print("Using momentum optimizer")
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum',
        use_nesterov=True)
  elif FLAGS.optimizer == 'rmsprop':
    if quantizer is None:
      optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.momentum,
        epsilon=FLAGS.opt_epsilon)
    else:
      optimizer = QRMSProp.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.momentum,
        epsilon=FLAGS.opt_epsilon,
        quantizer=quantizer)
  elif FLAGS.optimizer == 'sgd':
    if quantizer is None:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        optimizer = QSGD.GradientDescentOptimizer(learning_rate,
                                                quantizer=quantizer)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer

def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    with tf.Session() as sess:
      #######################
      # Quantizers          #
      #######################

      if FLAGS.intr_grad_quantizer is not '':
          qtype, qargs= utils.split_quantizer_str(FLAGS.intr_grad_quantizer)
          intr_grad_quantizer= utils.quantizer_selector(qtype, qargs)
      else:
          intr_grad_quantizer= None

      if FLAGS.extr_grad_quantizer is not '':
          qtype, qargs= utils.split_quantizer_str(FLAGS.extr_grad_quantizer)
          extr_grad_quantizer= utils.quantizer_selector(qtype, qargs)
      else:
          extr_grad_quantizer= None

      intr_q_map=utils.quantizer_map(FLAGS.intr_qmap)
      extr_q_map=utils.quantizer_map(FLAGS.extr_qmap)
      weight_q_map=utils.quantizer_map(FLAGS.weight_qmap)
      print("Intr QMap:%s"%intr_q_map)

      # Create global_step
      global_step = tf.train.create_global_step()

      ######################
      # Select the dataset #
      ######################
      dataset = dataset_factory.get_dataset(
          FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

      ######################
      # Select the network #
      ######################
      network_fn = nets_factory.get_network_fn(
          FLAGS.model_name,
          num_classes=(dataset.num_classes - FLAGS.labels_offset),
          weight_decay=FLAGS.weight_decay,
          is_training=True,
          intr_q_map=intr_q_map, extr_q_map=extr_q_map,
          weight_q_map=weight_q_map)

      ##############################################################
      # Create a dataset provider that loads data from the dataset #
      ##############################################################
      images = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 28, 28, 1])
      labels = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 10])
      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      ####################
      # Define the model #
      ####################
      logits, end_points = network_fn(images)

      # Specify the loss function #
      loss = tf.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels,
          label_smoothing=FLAGS.label_smoothing, weights=1.0)
      regularizer_losses = tf.add_n(tf.losses.get_regularization_losses())
      total_loss = loss + regularizer_losses


      # Gather initial summaries.
      summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by network_fn.
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

      #############
      # Summaries #
      #############

      # Add summaries for end_points.
      for end_point in end_points:
        x = end_points[end_point]
        summaries.add(tf.summary.histogram('activations/' + end_point, x))
        summaries.add(tf.summary.scalar('sparse_activations/' + end_point,
                                        tf.nn.zero_fraction(x)))

      # Add summaries for losses.
      for loss in tf.get_collection(tf.GraphKeys.LOSSES):
        summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
      summaries.add(tf.summary.scalar('losses_total', total_loss))
      summaries.add(tf.summary.scalar('losses_regularization', regularizer_losses))
      summaries.add(tf.summary.scalar('losses_classification', loss))

      #########################################
      # Configure the optimization procedure. #
      #########################################
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate, intr_grad_quantizer)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))


      # Variables to train.
      variables_to_train = _get_variables_to_train()

      # Create gradient updates.
      # quantize 'clones_gradients'
      clones_gradients = optimizer.compute_gradients(total_loss)
      if extr_grad_quantizer is not None:
          clones_gradients=[(extr_grad_quantizer.quantize(gv[0]),gv[1])
                              for gv in clones_gradients]

      # Add gradients to summary
      for gv in clones_gradients:
          summaries.add(tf.summary.histogram('gradient/%s'%gv[1].op.name, gv[0]))
          summaries.add(tf.summary.scalar('gradient-sparsity/%s'%gv[1].op.name,
                                      tf.nn.zero_fraction(gv[0])))

      grad_updates = optimizer.apply_gradients(clones_gradients,
                                               global_step=global_step)
      update_ops.append(grad_updates)

      update_op = tf.group(*update_ops)
      train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                        name='train_op')

      # Ensemble related ops
      variables_to_ensemble = {
              v.name:tf.Variable(tf.zeros_like(v)) for v in variables_to_train
      }
      ensemble_counter = tf.Variable(0.)
      variable_ensemble_ops = [
        tf.assign(variables_to_ensemble[v.name],
                 (variables_to_ensemble[v.name]*ensemble_counter + v)/(ensemble_counter + 1.)) \
        for v in variables_to_train
      ]
      ensemble_counter_update_op = tf.assign(ensemble_counter, ensemble_counter + 1)
      ensemble_replace_ops = [
        tf.assign(v, variables_to_ensemble[v.name]) for v in variables_to_train
      ]

      ##############################
      #  Evaluation pass  (Start)  #
      ##############################

      # Define the metrics:

      eval_pred= tf.squeeze(tf.argmax(logits, 1))
      eval_gtrs= tf.squeeze(tf.argmax(labels, 1))

      acc_value, acc_update = tf.metrics.accuracy(eval_pred, eval_gtrs)
      val_summary_lst = []
      val_summary_lst.append(tf.summary.scalar('val_acc', acc_value, collections=[]))
      val_summary_lst.append(tf.summary.scalar('val_err', 1-acc_value, collections=[]))
      val_summary_lst.append(
              tf.summary.scalar('val_err_perc', 100*(1-acc_value), collections=[]))
      val_summary = tf.summary.merge(val_summary_lst)

      num_batches = math.ceil(dataset.num_samples / (float(FLAGS.batch_size)) )

      # Merge all summaries together.
      summary_op = tf.summary.merge(list(summaries))

      train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

      ###########################
      # Kicks off the training. #
      ###########################
      sess.run(tf.global_variables_initializer())
      from tensorflow.examples.tutorials.mnist import input_data
      mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

      total_epoches = FLAGS.num_epoches
      for e in range(total_epoches):
        # Training pass
        sess.run(tf.local_variables_initializer())
        num_training_batches = 60000//FLAGS.batch_size
        for i in range(num_training_batches):
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            if i % FLAGS.log_every_n_steps == 0 and i > 0:
                summary_value, loss_value, acc = sess.run(
                  [summary_op, total_loss, acc_value],
                  feed_dict={
                    images : np.reshape(batch_xs, (FLAGS.batch_size, 28, 28, 1)),
                    labels : batch_ys
                })
                train_writer.add_summary(summary_value, i+e*num_training_batches)

                print("[%d/%d] loss %.5f err %.3f%%"\
                     %(i, num_training_batches, loss_value, (1. - acc)*100))
                sess.run(tf.local_variables_initializer())

            sess.run([update_op, acc_update], feed_dict={
                images : np.reshape(batch_xs, (FLAGS.batch_size, 28, 28, 1)),
                labels : batch_ys
            })

        # Validation pass
        sess.run(tf.local_variables_initializer())
        for i in range(10000//FLAGS.batch_size):
            batch_xs, batch_ys = mnist.validation.next_batch(FLAGS.batch_size)
            sess.run([acc_update], feed_dict={
                images : np.reshape(batch_xs, (FLAGS.batch_size, 28, 28, 1)),
                labels : batch_ys
            })
        val_acc, val_summary_value = sess.run([acc_value, val_summary], feed_dict={
          images : np.reshape(batch_xs, (FLAGS.batch_size, 28, 28, 1)),
          labels : batch_ys,
        })
        print("Epoch[%d/%d] : ValErr:%.3f%%"%(e, total_epoches, (1-val_acc)*100))
        train_writer.add_summary(val_summary_value, e)

        # Ensemble pass
        # if (e+1) % 5 == 0:
        if (e+1) % 20 == 0 :
            sess.run(variable_ensemble_ops)
            sess.run(ensemble_counter_update_op)
            print("Ensembled epoch %d weights"%e)


      # Validation Pass for Weight Ensembled Networks
      sess.run(tf.local_variables_initializer())
      sess.run(ensemble_replace_ops)
      for i in range(10000//FLAGS.batch_size):
          batch_xs, batch_ys = mnist.validation.next_batch(FLAGS.batch_size)
          sess.run([acc_update], feed_dict={
              images : np.reshape(batch_xs, (FLAGS.batch_size, 28, 28, 1)),
              labels : batch_ys
          })
      val_acc, val_summary_value = sess.run([acc_value, val_summary], feed_dict={
        images : np.reshape(batch_xs, (FLAGS.batch_size, 28, 28, 1)),
        labels : batch_ys,
      })
      print("Ensembled network ValErro:%.3f%%"%((1-val_acc)*100))
      train_writer.add_summary(val_summary_value, total_epoches)

if __name__ == '__main__':
  tf.app.run()
