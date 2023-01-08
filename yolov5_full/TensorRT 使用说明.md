# **TensorRT 使用说明**



# **安装**

CUDA 11.4:  https://developer.nvidia.com/cuda-11-4-4-download-archive

cudnn v8.2.4 for CUDA 11.4:  https://developer.nvidia.com/rdp/cudnn-archive

TensorRT 8.2.5（需下载tar格式，解压即可）:  https://developer.nvidia.com/nvidia-tensorrt-8x-download



安装完成后，需添加环境变量，可执行文件添加到 PATH 中，动态链接库添加到 LD_LIBRARY_PATH 中



注意:

1）TensorRT依赖对应版本的 CUDA、cudnn，请按上述版本下载

2）TensorRT本质上是闭源软件，无需再从其 GitHub 网站下载其他资源



# **转换**

下面以YOLOv5（v6.1 版本）为例进行讨论

## **获取模型文件**



```
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.1  
wget -c https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
cat models/yolov5s.yaml
```





## **PyTorch 转 ONNX**



```
python export.py --include onnx --dynamic --simplify
```





## **ONNX 务必测试**



```
import onnxruntime as ort
import numpy as np
x = np.random.rand(1, 3, 640, 640).astype(np.float32)
ort_sess = ort.InferenceSession('yolov5s.onnx')
outputs = ort_sess.run(None, {'images': x})
for item in outputs:
    print(item.shape)
```





如果需要，在这里可以可视化 ONNX 模型：https://netron.app/



## **转 TensorRT（重点）**

1. 下面 images 是输入节点名称，导出 ONNX 模型的Python代码中设置的，如果设置为input或其他名称，此处也要进行修改
2. minShapes表示最小尺度，maxShapes表示最大尺度，optShapes表示介于之间的某个常见尺度。如果模型是固定尺度，或者不想使用这个动态功能，直接设置为相同数值即可，如这里就是这样
3. --fp16 必须加上，这样模型用半精度推理，速度更快



```
trtexec --onnx=yolov5s.onnx \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:4x3x640x640 \
        --maxShapes=images:8x3x640x640 \
        --workspace=4096 \
        --saveEngine=yolov5s.engine \
        --fp16
```





注意, ONNX 模型输出的 outputs 顺序, 和 TensorRT 优化过的顺序不一定相同, 编写解码代码前, 建议打印 TensorRT 的各输出 shape 进行确认。

例如, 这里 ONNX 输出 shape:

\```

(1, 25200, 85)

(1, 3, 80, 80, 85)

(1, 3, 40, 40, 85)

(1, 3, 20, 20, 85)

\```

TensorRT 输出 shape:

\```

2, 3, 80, 80, 85

2, 3, 40, 40, 85

2, 3, 20, 20, 85

2, 25200, 85

\```



# **调用**

完成 PyTorch => ONNX => TensorRT 格式转换，且全过程没有报错之后，就可以写代码进行调用了。

下面是示例工程，按其中 README 操作即可。



链接: https://pan.baidu.com/s/16FVYfUAlnWszs2CU4cxr1Q 提取码: 1t68 



# **参考**

https://github.com/RichardoMrMu/yolov5-deepsort-tensorrt/blob/main/yolo/src/yolov5_lib.cpp