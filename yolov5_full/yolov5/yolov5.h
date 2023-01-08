#ifndef __YOLOV5_H__
#define __YOLOV5_H__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

static const int CLASS_NUM = 80;
static const int INPUT_SIZE = 640;
static const float NMS_THRESH = 0.4;
static const float CONF_THRESH = 0.5;

typedef struct {
    int x;
    int y;
    int w;
    int h;
    float conf;
    int class_id;
    std::string class_name;
} BoxItem;

// 将图像转换成归一化 float vector
int get_norm_data(cv::Mat &img, std::vector<float> &result, int img_size, float mean=0.f, float scale=1/255.f, int fill_color=114);

// YOLOv5 decoding for multi-heads, such as:
// (1, 3, 80, 80, 85)
// (1, 3, 40, 40, 85)
// (1, 3, 20, 20, 85)
std::vector<std::vector<BoxItem>> yolov5_decoding(
    const std::vector<int> &raw_heights,         // 每张图片的高度
    const std::vector<int> &raw_widths,          // 每张图片的宽度
    const std::vector<std::vector<float>> &data, // YOLO可能有3+个输出Tensor, 这里依次存放每个输出
    const std::vector<std::vector<int>> &shape   // 这是对应每个输出的尺度, 元素格式: batch_size x anchors_each_grp x grid_size x grid_size x (5 + class_num)
);

#endif // __YOLOV5_H__
