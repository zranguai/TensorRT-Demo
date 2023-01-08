#ifndef __TRT_H__
#define __TRT_H__

#include <string>
#include <vector>

typedef struct {
    std::vector<float> data;
    std::vector<int> shape;
} Tensor;

// Return 0 on success
int init_trt(const char *model_path);

// Return 0 on success
int trt_inference(const std::vector<Tensor> &input, std::vector<Tensor> &output);

void release_trt();

#endif
