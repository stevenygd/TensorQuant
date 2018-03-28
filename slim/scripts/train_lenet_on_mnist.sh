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
TRAIN_DIR=./tmp/lenet-model/base

# TRAIN_DIR=./tmp/lenet-model/intr_grad_stochastic_16_14
# TRAIN_DIR=./tmp/lenet-model/intr_grad_stochastic_16_10
# TRAIN_DIR=./tmp/lenet-model/intr_grad_stochastic_16_8

# TRAIN_DIR=./tmp/lenet-model/extr_grad_stochastic_16_14
# TRAIN_DIR=./tmp/lenet-model/extr_grad_stochastic_16_10
# TRAIN_DIR=./tmp/lenet-model/extr_grad_stochastic_16_8

# TRAIN_DIR=./tmp/lenet-model/extr_grad_stochastic_16_8_ensemble # 0.97930 -> 0.98270
# TRAIN_DIR=./tmp/lenet-model/extr_grad_stochastic_16_8_ensemble_100
# TRAIN_DIR=./tmp/lenet-model/extr_grad_stochastic_8_6_ensemble_100

# TRAIN_DIR=./tmp/lenet-model/extr_grad_nearest_16_14
# TRAIN_DIR=./tmp/lenet-model/extr_grad_nearest_16_10
# TRAIN_DIR=./tmp/lenet-model/extr_grad_nearest_16_8

# TRAIN_DIR=./tmp/lenet-model/extr_grad_layer_stochastic_16_14

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
  --num_epoches=100 \
  --dataset_name=mnist \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet \
  --preprocessing_name=lenet \
  --max_number_of_steps=10000 \
  --batch_size=100 \
  --learning_rate=0.1 \
  --save_interval_secs=600 \
  --save_summaries_secs=10 \
  --log_every_n_steps=100 \
  --optimizer=momentum \
  --momentum=0.9 \
  --learning_rate_decay_type=exponential \
  --learning_rate_decay_factor=0.95 \
  --weight_decay=0;
  # --extr_grad_quantizer=stochastic,8,6;


