#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/action_recognition/log \
    mpirun -np 4 \
    cmake_build/install/bin/caffe train \
    --solver=models/action_recognition/vgg_16_rgb_solver.prototxt \
    --weights=vgg_16_action_rgb_pretrain.caffemodel

