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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

import utils

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.client import device_lib

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 16,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')


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

###############################
# Output File
###############################
tf.app.flags.DEFINE_string(
    'output_file', None, 'File in which metrics output is safed.'
    )

tf.app.flags.DEFINE_string(
    'comment', '', 'Optional comment for entry in output file.'
    )

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO) #can be WARN or INFO
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ########################
    # Determine Quantizers #
    ########################
    intr_q_map=utils.quantizer_map(FLAGS.intr_qmap)
    extr_q_map=utils.quantizer_map(FLAGS.extr_qmap)
    weight_q_map=utils.quantizer_map(FLAGS.weight_qmap)

    ####################
    # Select the model #
    ####################
    if 'resnet' in FLAGS.model_name:
        labels_offset=1
    else:
        labels_offset=0
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - labels_offset),
        is_training=False,
        intr_q_map=intr_q_map, extr_q_map=extr_q_map,
        weight_q_map=weight_q_map)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=8 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size*4)
    [image, label] = provider.get(['image', 'label'])
    label -= labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    ####################
    # Define the model #
    ####################
    start_time_build = time.time()

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    logits, endpoints = network_fn(images)
    predictions= tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    #tf.logging.info('Number of parameters per layer and endpoint:')
    #for var in endpoints:
    #    tf.logging.info('%s: %d'%(var,utils.count_trainable_params(var)))
    #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))


    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        #'Recall_5': slim.metrics.streaming_recall_at_k(logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches and FLAGS.max_num_batches > 0:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      #num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
      num_batches = math.ceil(dataset.num_samples / (float(FLAGS.batch_size)) )

    # get checkpoint
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path
    if checkpoint_path is None:
        raise ValueError('No Checkpoint found!')
    tf.logging.info('Evaluating %s' % checkpoint_path)

    # get per layer number of variables
    weights_list_param_count = utils.get_variables_count_dict('weights')
    biases_list_param_count = utils.get_variables_count_dict('biases')

    # get ops calculating the layerwise and overall sparsity of weights
    weights_layerwise_sparsity_op, weights_overall_sparsity_op = (
                utils.get_sparsity_ops('weights') )
    biases_layerwise_sparsity_op, biases_overall_sparsity_op = (
                utils.get_sparsity_ops('biases') )

    # get ops for activation sparsity
    activation_layerwise_sparsity = {}
    for key in endpoints.keys():
      activation_layerwise_sparsity[key]=tf.nn.zero_fraction(endpoints[key])

    '''
    # get the quantized weight tensors for sparsity estimation
    weights_name_list, weights_list = utils.get_variables_list('weights')
    biases_name_list, biases_list = utils.get_variables_list('biases')

    # count number of elements in each layer
    weights_list_param_count = [ utils.get_nb_params_shape(x.get_shape())
                                    for x in weights_list]
    weights_list_param_count = dict(zip(weights_name_list, weights_list_param_count))
    biases_list_param_count = [ utils.get_nb_params_shape(x.get_shape())
                                    for x in biases_list]
    biases_list_param_count = dict(zip(biases_name_list, biases_list_param_count))

    # get zero fraction for each layer
    weights_layerwise_sparsity_op=[ tf.nn.zero_fraction(x) for x in weights_list]
    biases_layerwise_sparsity_op=[ tf.nn.zero_fraction(x) for x in biases_list]

    # add overall weights sparsity to summary
    weights_overall_sparsity_op=[ tf.reshape(x,[tf.size(x)]) for x in weights_list]
    weights_overall_sparsity_op=tf.concat(weights_overall_sparsity_op,axis=0)
    #l_weights_values = weights_overall_sparsity_op # a list of all weight values, for histogram
    summary_name = 'eval/weight_overall_sparsity'
    weights_overall_sparsity_op=tf.nn.zero_fraction(weights_overall_sparsity_op)
    op = tf.summary.scalar(summary_name, weights_overall_sparsity_op, collections=[])
    #op = tf.Print(op, [value], summary_name)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # add overall bias sparsity to summary
    biases_overall_sparsity_op=[ tf.reshape(x,[tf.size(x)]) for x in biases_list]
    biases_overall_sparsity_op=tf.concat(biases_overall_sparsity_op,axis=0)
    summary_name = 'eval/biases_overall_sparsity'
    biases_overall_sparsity_op=tf.nn.zero_fraction(biases_overall_sparsity_op)
    op = tf.summary.scalar(summary_name, biases_overall_sparsity_op, collections=[])
    #op = tf.Print(op, [value], summary_name)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    '''

    # Run Session
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    #config.gpu_options.allocator_type = 'BFC'
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.7

    # Final ops, used for statistics
    final_op = (list(names_to_values.values()),
                       weights_layerwise_sparsity_op, biases_layerwise_sparsity_op,
                       weights_overall_sparsity_op, biases_overall_sparsity_op,
                       list(activation_layerwise_sparsity.values()) )
    print('Running %s for %d iterations'%(FLAGS.model_name,num_batches))
    start_time_simu = time.time()
    run_values = slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        final_op=final_op,
        variables_to_restore=variables_to_restore,
        session_config=config)
    runtime=time.time()-start_time_simu
    buildtime=start_time_simu-start_time_build
    accuracy = run_values[0][0]
    weight_sparsity_values = run_values[1]
    bias_sparsity_values = run_values[2]
    weight_sparsity=run_values[3]
    bias_sparsity=run_values[4]
    activation_layerwise_sparsity=dict(zip( list(activation_layerwise_sparsity.keys()),
                                            [float(x) for x in run_values[5]] ))

    # compute sparsity
    print('Calculating sparsity...')
    weight_sparsity_layerwise = {}
    weights_name_list = list(weights_list_param_count.keys())
    for it in range(len(weights_name_list)):
        name=weights_name_list[it]
        weight_sparsity_layerwise[name] = float(weight_sparsity_values[it])
    bias_sparsity_layerwise = {}
    biases_name_list = list(biases_list_param_count.keys())
    for it in range(len(biases_name_list)):
        name=biases_name_list[it]
        bias_sparsity_layerwise[name] = float(bias_sparsity_values[it])

    # print statistics
    print('\nStatistics:')
    print('Accuracy: %.2f%%'%(accuracy*100))
    print('Buildtime: %f sec'%buildtime)
    print('Runtime: %f sec'%runtime)
    print('---')
    print('Weight sparsity: %f%%'%(weight_sparsity*100))
    print('Layerwise weight sparsity:')
    for key in weight_sparsity_layerwise.keys():
        print("     %s: %.2f%%"%(key, weight_sparsity_layerwise[key]*100))
    print('Bias sparsity: %f%%'%(bias_sparsity*100))
    print('Layerwise bias sparsity:')
    for key in bias_sparsity_layerwise.keys():
        print("     %s: %.2f%%"%(key, bias_sparsity_layerwise[key]*100))
    print('---')
    print('Activation sparsity:')
    for key in activation_layerwise_sparsity.keys():
        print("     %s: %.2f%%"%(key, activation_layerwise_sparsity[key]*100))
    print('Comment: %s'%(FLAGS.comment))

    tf.train.export_meta_graph(filename=FLAGS.checkpoint_path+'/model.meta')

    # write data to .json file
    if FLAGS.output_file is not None:
        if os.path.exists(FLAGS.output_file) == False:
            json_data = []
        else:
            with open(FLAGS.output_file) as hfile:
                json_data = json.load(hfile)
        new_data={}
        new_data["accuracy"]=accuracy.tolist()
        new_data["net"]=FLAGS.model_name
        new_data["samples"]=(num_batches*FLAGS.batch_size)
        new_data["weight_sparsity"]=weight_sparsity.tolist()
        new_data["weight_sparsity_layerwise"]=weight_sparsity_layerwise
        new_data["weight_count_layerwise"]=weights_list_param_count
        new_data["bias_sparsity"]=bias_sparsity.tolist()
        new_data["bias_sparsity_layerwise"]=bias_sparsity_layerwise
        new_data["bias_count_layerwise"]=biases_list_param_count
        new_data["activation_sparsity_layerwise"]=activation_layerwise_sparsity
        new_data["runtime"]=runtime
        new_data["comment"]=FLAGS.comment

        json_data.append(new_data)
        with open(FLAGS.output_file,'w') as hfile:
            json.dump(json_data, hfile)

    print('Done.')
if __name__ == '__main__':
  tf.app.run()
