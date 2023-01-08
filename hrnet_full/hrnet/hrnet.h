#ifndef __HRNET_H__
#define __HRNET_H__

#include <opencv2/opencv.hpp>

const int INPUT_WIDTH = 192;
const int INPUT_HEIGHT = 256;

// 单图的归一化
int get_norm_data(
    const cv::Mat &img,         // （可能包含目标的）图片, 只支持一个人
    std::vector<float> &result, // 归一化结果
    std::vector<float> &center, // 中心点, 解码需要
    std::vector<float> &scale   // 缩放系数, 解码需要
);

// 批量解码（batch）
int hrnet_decoding(
    const std::vector<float> &heatmap,               // 模型推理输出的数据, 只有一层 vector, 因为输入、输出都只有 1 路
    const std::vector<int> &shapes,                  // 模型推理输出的 shape
    const std::vector<std::vector<float> > &centers, // 每个图的中心点
    const std::vector<std::vector<float> > &scales,  // 每个图的缩放尺度
    std::vector<std::vector<cv::Point> > &points     // 存放结果
);

#endif // __HRNET_H__
