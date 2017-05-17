#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/semantic_segmentation/log \
    mpirun -np 8 \
    cmake_build/install/bin/caffe train \
    --solver=models/semantic_segmentation/fcn_vgg_16_solver.prototxt \
    --weights=VGG_ILSVRC_16_layers_conv.caffemodel