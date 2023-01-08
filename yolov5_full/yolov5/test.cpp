#include "../comm/trt.h"
#include "yolov5.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>

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
    std::vector<cv::Mat> imgs; // 保存图片方便后面解码使用尺度信息及画图

    for (size_t n = 0; n < img_paths.size(); n++) {
        cv::Mat img = cv::imread(img_paths[n]);

        if (img.empty()) {
            std::cerr << "read image error" << std::endl;
            return -1;
        }

        imgs.push_back(img);
        std::vector<float> norm_data;

        if (get_norm_data(img, norm_data, INPUT_SIZE) != 0) {
            std::cerr << "norm error" << std::endl;
            return -1;
        }

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
    std::vector<int> raw_heights;
    std::vector<int> raw_widths;
    std::vector<std::vector<float>> data;
    std::vector<std::vector<int>> shape;

    for (auto &im: imgs) {
        raw_heights.push_back(im.rows);
        raw_widths.push_back(im.cols);
    }

    // 注意, 此处解码实现只使用了 N=3 个分离的支路输出, 不需要 (N, 25200, 85) 这个汇总的支路
    // 根据上述 print shape 信息得知, 忽略最后一个支路输出
    for (size_t i = 0; i < output.size(); i++) {
        if (output.size() - 1 == i) {
            continue;
        }

        auto item = output[i];
        data.push_back(std::move(item.data)); // 使用 std::move 加速, 因为 output 这个变量下面不需要使用了
        shape.push_back(std::move(item.shape));
    }

    std::vector<std::vector<BoxItem>> batch_boxes = yolov5_decoding(raw_heights, raw_widths, data, shape);
    std::cout << "decoding ok" << std::endl;

    // 可以画图看效果
    std::vector<int> comp_params;
    comp_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    comp_params.push_back(100);

    for (size_t i = 0; i < batch_boxes.size(); i++) {
        auto boxes = batch_boxes[i];
        std::cout << "image index: " << i << ", path: " << img_paths[i] << ", num of boxes: " << boxes.size() << std::endl;

        if (boxes.size() > 0) {
            for (size_t j = 0; j < boxes.size(); j++) {
                BoxItem box = boxes[j];
                cv::Rect r(box.x, box.y, box.w, box.h);
                std::string info = std::to_string(box.class_id) + ": " + std::to_string(box.conf).substr(0, 4);
                cv::rectangle(imgs[i], r, cv::Scalar(0, 0, 255), 2);
                cv::putText(imgs[i], info, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255), 2);
            }

            std::string save_path = "output_" + std::to_string(i) + ".jpg";
            cv::imwrite(save_path, imgs[i], comp_params);
        }
    }

    release_trt();
    return 0;
}
