如果没有安装 OpenCV:
    从 https://opencv.org/releases/ 下载opencv-3.4.16.zip

    sudo apt update
    sudo apt install -y cmake g++ wget unzip cmake build-essential \
                        libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 \
                        libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

    sudo mkdir -p /opt/tools
    cd opencv-3.4.16
    mkdir release && cd release
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/opt/tools/opencv-3.4.16 ..
    make -j8
    sudo make install

    参考了：https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html

步骤：
    1. 安装 CUDA、cudnn、TensorRT 软件，设置对应的环境变量
    2. 获取示例 YOLOv5 模型文件，按说明转成 ONNX, 再转成 TensorRT 格式

    3. 修改 Makefile 中 CUDA_ROOT、TRT_ROOT、OPENCV_ROOT 为自己的实际路径
    4. 修改 run.sh 中 CUDA_ROOT、TRT_ROOT、OPENCV_ROOT 为自己的实际路径
    5. 执行 bash run.sh 查看结果

    6. 自己修改 test.cpp，完整地完成一个例子，如 YOLOv5 或其他分类、检测模型
    7. 执行 bash run.sh 查看结果
