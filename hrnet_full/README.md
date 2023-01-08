# HRNet v2 说明
## GitHub 主页
https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

## 编译依赖库
根据主页上的说明:
```
git clone https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git
cd deep-high-resolution-net.pytorch
cd lib && make
```

## 转 ONNX
将提供的 convert.py 放在 tools 目录下, 下载作者提供的模型 pose_hrnet_w32_256x192.pth 照作者要求放在 models/pytorch/pose_coco 目录下
```
python tools/convert.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
    TEST.USE_GT_BBOX False
```
如果报错, pip 安装缺失的库

## ONNX 测试
```
import onnxruntime as ort
import numpy as np
x = np.random.rand(1, 3, 256, 192).astype(np.float32)
ort_sess = ort.InferenceSession('hrnet.onnx')
outputs = ort_sess.run(None, {'input': x})
for item in outputs:
    print(item.shape)
```
期望输出 (1, 17, 64, 48)

## 转 TensorRT
```
trtexec --onnx=hrnet.onnx \
        --minShapes=input:1x3x256x192 \
        --optShapes=input:4x3x256x192 \
        --maxShapes=input:8x3x256x192 \
        --workspace=4096 \
        --saveEngine=hrnet.engine \
        --fp16
```

## 测试
使用本文提供的脚本前, 记得先修改为自己的环境:
- 修改 Makefile 中 CUDA_ROOT、TRT_ROOT、OPENCV_ROOT 为自己的实际路径
- 修改 run.sh 中 CUDA_ROOT、TRT_ROOT、OPENCV_ROOT 为自己的实际路径
- 执行 bash run.sh 查看结果图片
