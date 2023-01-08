#include "../comm/trt.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>

static const int INPUT_SIZE = 640;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage:\n\t./test [engine path] [image path 1] ..." << std::endl;
        return 1;
    }

    const char *model_path = argv[1];
    std::vector<std::string> img_paths;

    // 读取各个图片路径
    for (int i = 0; i < argc - 2; i++) {
        img_paths.push_back(argv[2 + i]);
    }

    // 构造输入，假设输入只有1个head, batch size就是图片数
    Tensor inp0;
    inp0.shape.push_back(int(img_paths.size()));
    inp0.shape.push_back(3);
    inp0.shape.push_back(INPUT_SIZE);
    inp0.shape.push_back(INPUT_SIZE);

    // 初始化
    if (init_trt(model_path) != 0) {
        std::cerr << "init failed" << std::endl;
        return -1;
    }

    // 开始构造输入数据
    std::vector<cv::Mat> imgs;

    for (size_t n = 0; n < img_paths.size(); n++) {
        cv::Mat img = cv::imread(img_paths[n]);
        imgs.push_back(img);
        std::vector<float> norm_data;
        norm_data.resize(1 * 3 * INPUT_SIZE, INPUT_SIZE);

        // 此次进行归一化，包括 letter box 操作，结果放在 norm_data 中
        // ...
        
        for (auto x: norm_data) {
            inp0.data.push_back(x);
        }
    }

    // 输入数据
    std::vector<Tensor> input;
    input.push_back(inp0);

    // 存放输出结果
    std::vector<Tensor> output;

    // 进行推理
    if (trt_inference(input, output) != 0) {
        std::cerr << "trt inference failed" << std::endl;
        return -1;
    }

    // 打印各个输出 heads 的 shape
    std::cout << "output shape: " << std::endl;

    for (size_t c = 0; c < output.size(); c++) {
        std::cout << "    head " << c << " shape: ";

        for (size_t d = 0; d < output[c].shape.size(); d++) {
            std::cout << output[c].shape[d];

            if (d < output[c].shape.size() - 1) {
                std::cout << ", ";
            }
        }

        std::cout << std::endl;
    }

    // 此处进行模型解码，从输出Tensor中（可能是多个heads）还原出坐标等信息
    // ...

    // 可以画图看效果
    // ...

    release_trt();
    return 0;
}
