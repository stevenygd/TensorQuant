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

TRAIN_DIR=./tmp/lenet-model/intr_grad_stochastic_16_14 # Accuracy: 98.04%
# TRAIN_DIR=./tmp/lenet-model/intr_grad_stochastic_16_10 # Accuracy: 95.74%
# TRAIN_DIR=./tmp/lenet-model/intr_grad_stochastic_16_8 # Accuracy: 94.46%

# TRAIN_DIR=./tmp/lenet-model/extr_grad_stochastic_16_14 # Accuracy: 98.08%
# TRAIN_DIR=./tmp/lenet-model/extr_grad_stochastic_16_10 # Accuracy: 98.13%
# TRAIN_DIR=./tmp/lenet-model/extr_grad_stochastic_16_8 # Accuracy: 98.00%

# TRAIN_DIR=./tmp/lenet-model/intr_grad_nearest_16_14 # Accuracy: 98.01%
# TRAIN_DIR=./tmp/lenet-model/intr_grad_nearest_16_10 # Accuracy: 97.17%
# TRAIN_DIR=./tmp/lenet-model/intr_grad_nearest_16_8  # Accuracy: 89.79%


rm -r ${TRAIN_DIR}
mkdir ${TRAIN_DIR}

# Where the dataset is saved to.
DATASET_DIR=~/tmp/mnist

# Download the dataset
#python download_and_convert_data.py \
#  --dataset_name=mnist \
#  --dataset_dir=${DATASET_DIR}

# Run training.
export CUDA_VISIBLE_DEVICES=0
export EVAL_INTERVALS=1000
for i in `seq 1 5`; do
    python train_image_classifier.py \
      --train_dir=${TRAIN_DIR} \
      --dataset_name=mnist \
      --dataset_split_name=train \
      --dataset_dir=${DATASET_DIR} \
      --model_name=lenet \
      --preprocessing_name=lenet \
      --max_number_of_steps=$EVAL_INTERVALS \
      --batch_size=50 \
      --learning_rate=0.01 \
      --save_interval_secs=600 \
      --save_summaries_secs=10 \
      --log_every_n_steps=100 \
      --optimizer=sgd \
      --learning_rate_decay_type=fixed \
      --weight_decay=0 \
      --intr_grad_quantizer=stochastic,16,8;

    # Run evaluation.
    python eval_image_classifier.py \
      --checkpoint_path=${TRAIN_DIR} \
      --eval_dir=${TRAIN_DIR} \
      --dataset_name=mnist \
      --dataset_split_name=test \
      --dataset_dir=${DATASET_DIR} \
      --model_name=lenet;
done
