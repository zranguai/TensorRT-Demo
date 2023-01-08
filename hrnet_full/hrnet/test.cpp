#include "../comm/trt.h"
#include "hrnet.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage:\n\t./test [engine path] [image path 1] [image path 2] ..." << std::endl;
        return 1;
    }

    const char *model_path = argv[1];
    std::vector<std::string> img_paths;

    // 读取各个图片路径
    for (int i = 0; i < argc - 2; i++) {
        img_paths.push_back(argv[2 + i]);
    }

    // 构造输入，假设输入只有 1 个 head, batch size 就是图片数
    Tensor inp0;
    inp0.shape.push_back(int(img_paths.size()));
    inp0.shape.push_back(3);
    inp0.shape.push_back(INPUT_HEIGHT);
    inp0.shape.push_back(INPUT_WIDTH);
    
    // 初始化
    if (init_trt(model_path) != 0) {
        std::cerr << "init failed" << std::endl;
        return -1;
    }

    // 开始构造输入数据
    std::vector<cv::Mat> imgs; // 保存图片方便后面画图
    std::vector<std::vector<float> > centers;
    std::vector<std::vector<float> > scales;

    for (size_t i = 0; i < img_paths.size(); i++) {
        cv::Mat img = cv::imread(img_paths[i]);

        if (img.empty()) {
            std::cerr << "read image error" << std::endl;
            return -1;
        }

        imgs.push_back(img);
        std::vector<float> norm_data;
        std::vector<float> center;
        std::vector<float> scale;

        if (get_norm_data(img, norm_data, center, scale) != 0) {
            std::cerr << "norm error" << std::endl;
            return -1;
        }

        for (auto x: norm_data) {
            inp0.data.push_back(x);
        }

        centers.push_back(center);
        scales.push_back(scale);
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

    // 此处进行模型解码，从输出Tensor中还原出坐标等信息
    auto out0 = output[0];
    std::vector<std::vector<cv::Point> > points;
    
    if (hrnet_decoding(out0.data, out0.shape, centers, scales, points) != 0) {
        std::cerr << "pred error" << std::endl;
        return -1;
    }
    else {
        std::cout << "pred ok" << std::endl;
    }

    // 可以画图看效果
    std::vector<int> comp_params;
    comp_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    comp_params.push_back(100);

    for (size_t i = 0; i < points.size(); i++) {
        for (size_t j = 0; j < points[0].size(); j++) {
            cv::circle(imgs[i], cv::Point(points[i][j].x, points[i][j].y), 3, cv::Scalar(0, 0, 255), 3);

            std::string save_path = "output_" + std::to_string(i) + ".jpg";
            cv::imwrite(save_path, imgs[i], comp_params);
        }
    }

    std::cout << "saved" << std::endl;

    release_trt();
    return 0;
}
