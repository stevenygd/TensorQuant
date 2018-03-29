#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the MNIST dataset
# 2. Trains a LeNet model on the MNIST training set.
# 3. Evaluates the model on the MNIST testing set.
#
# Usage:
# cd slim
# ./slim/scripts/train_lenet_on_mnist.sh

# Where the checkpoint and logs will be saved to.

# TRAIN_DIR=./tmp/lenet-model/test
# TRAIN_DIR=./tmp/lenet-model/base
# TRAIN_DIR=./tmp/lenet-model/intr_grad_16_14_extr_layers_16_10
# TRAIN_DIR=./tmp/lenet-model/intr_grad_16_12_extr_layers_16_10
# TRAIN_DIR=./tmp/lenet-model/intr_grad_16_10_extr_layers_16_10

# TRAIN_DIR=./tmp/lenet-model/intr_grad_16_14_extr_others_16_14_extr_conv_16_10

# TRAIN_DIR=./tmp/lenet-model/intr_grad_16_14_extr_layers_16_10_nearest
TRAIN_DIR=./tmp/lenet-model/intr_grad_16_14_intr_layers_16_10_nearest

rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# # Download the dataset
# python download_and_convert_data.py \
#   --dataset_name=mnist \
#   --dataset_dir=${DATASET_DIR}

# Run training.

python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --num_epoches=200 \
  --dataset_name=mnist \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet \
  --preprocessing_name=lenet \
  --batch_size=100 \
  --learning_rate=0.1 \
  --save_interval_secs=600 \
  --save_summaries_secs=10 \
  --log_every_n_steps=100 \
  --optimizer=momentum \
  --momentum=0.9 \
  --learning_rate_decay_type=exponential \
  --learning_rate_decay_factor=0.95 \
  --weight_decay=0.0005 \
  --intr_grad_quantizer=nearest,16,14 \
  --intr_qmap=./tmp/lenet-model/QMaps/extr_qmap.json;


