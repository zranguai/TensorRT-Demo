#!/bin/bash

# 修改为自己的路径
CUDA_ROOT=/usr/local/cuda-11.3
TRT_ROOT=/opt/tools/TensorRT-8.2.3.0
OPENCV_ROOT=/opt/tools/opencv-3.4.16

export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TRT_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENCV_ROOT/lib:$LD_LIBRARY_PATH

echo "building..."
make -j4 && echo "build ok"

echo "start..."
./test yolov5s.engine ./images/bus.jpg
