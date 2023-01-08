#include "trt.h"
#include "logging.h"
#include "cuda_utils.h"
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include <fstream>
#include <iterator>

static Logger gLogger;
static int gBindNum;
static nvinfer1::IExecutionContext *gCtx;
static nvinfer1::ICudaEngine *gEngine;
static nvinfer1::IRuntime *gRuntime;

int init_trt(const char *model_path) {
    std::ifstream fs(model_path, std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(fs), {});
    fs.close();

    gRuntime = nvinfer1::createInferRuntime(gLogger);
    gEngine = gRuntime->deserializeCudaEngine((void *)buffer.data(), buffer.size());
    gCtx = gEngine->createExecutionContext();
    gBindNum = gEngine->getNbBindings();
    return 0;
}

int trt_inference(const std::vector<Tensor> &input, std::vector<Tensor> &output) {
    std::vector<void *> deviceBuffer;
    int numInpItems = input[0].shape[0] * input[0].shape[1] * input[0].shape[2] * input[0].shape[3];
    deviceBuffer.push_back(NULL);
    CUDA_CHECK(cudaMalloc(&deviceBuffer[0], numInpItems * sizeof(float)));
    nvinfer1::Dims4 inpDims = { input[0].shape[0], input[0].shape[1], input[0].shape[2], input[0].shape[3] };
    gCtx->setBindingDimensions(0, inpDims);

    for (int i = 1; i < gBindNum; i++) {
        deviceBuffer.push_back(NULL);
        nvinfer1::Dims numOutDims = gCtx->getBindingDimensions(i);
        int numOutItems = 1;
        for (int i = 0; i < numOutDims.nbDims; i++) { numOutItems *= numOutDims.d[i]; }
        CUDA_CHECK(cudaMalloc(&deviceBuffer[i], numOutItems * sizeof(float)));
    }
    
    CUDA_CHECK(cudaMemcpy(deviceBuffer[0], (void *)(&(input[0].data[0])), numInpItems * sizeof(float), cudaMemcpyHostToDevice));
    if (!gCtx->executeV2(deviceBuffer.data())) { return -1; }

    for (int i = 1; i < gBindNum; i++) {
        Tensor tensor;
        int numOutItems = 1;
        nvinfer1::Dims numOutDims = gCtx->getBindingDimensions(i);

        for (int j = 0; j < numOutDims.nbDims; j++) {
            int d = numOutDims.d[j];
            tensor.shape.push_back(d);
            numOutItems *= d;
        }

        tensor.data.resize(numOutItems);
        CUDA_CHECK(cudaMemcpy((void *)(&(tensor.data[0])), deviceBuffer[i], numOutItems * sizeof(float), cudaMemcpyDeviceToHost));
        output.push_back(tensor);
    }

    for (int i = 0; i < gBindNum; i++) { CUDA_CHECK(cudaFree(deviceBuffer[i])); }
    return 0;
}

void release_trt() {
    gCtx->destroy();
    gEngine->destroy();
    gRuntime->destroy();
}
