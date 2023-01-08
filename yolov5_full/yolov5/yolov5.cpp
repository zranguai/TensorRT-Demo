#include "yolov5.h"

#include <string.h>
#include <stdint.h>
#include <math.h>

static const size_t MAX_OUTPUT_BBOX_COUNT = 1000; // 设定最大 box 数目

// 3 组 anchors 的情况, YOLOv5 有些配置有 4 组
static const std::vector<int> strides = { 8, 16, 32 };
static const std::vector<std::vector<int>> anchors = {
    { 10,13, 16,30, 33,23 },     // P3/8
    { 30,61, 62,45, 59,119 },    // P4/16
    { 116,90, 156,198, 373,326 } // P5/32
};

static inline float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

static cv::Mat letter_box(const cv::Mat &img, int img_size, int fill_color) {
    int h = img.rows;
    int w = img.cols;
    const auto value = cv::Scalar(fill_color, fill_color, fill_color);

    if (h < img_size && w < img_size) {
        int top = (img_size - h) / 2;
        int bottom = img_size - h - top;
        int left = (img_size - w) / 2;
        int right = img_size - w - left;

        cv::Mat out;
        cv::copyMakeBorder(img, out, top, bottom, left, right, cv::BORDER_CONSTANT, value);
        return out;
    }
    else {
        float r_h = (float)img_size / h;
        float r_w = (float)img_size / w;

        if (r_w < r_h) {
            w = img_size;
            h = int(r_w * h);

            if (h % 2 != 0) {
                h--;
            }
        }
        else {
            h = img_size;
            w = int(r_h * w);

            if (w % 2 != 0) {
                w--;
            }
        }

        cv::Mat t, out;
        cv::resize(img, t, cv::Size(w, h));
        const auto value = cv::Scalar(0, 0, 0);
        cv::copyMakeBorder(t, out, (img_size - h) / 2, (img_size - h) / 2, (img_size - w) / 2, 
            (img_size - w) / 2, cv::BORDER_CONSTANT, value);
        return out;
    }
}

int get_norm_data(cv::Mat &img, std::vector<float> &result, int img_size, float mean, float scale, int fill_color) {
    if (img.empty() || img.channels() != 3) {
        return -1;
    }

    if (img.rows != img_size || img.cols != img_size) {
        img = letter_box(img, img_size, fill_color);
    }

    int rows = img.rows;
    int cols = img.cols;
    result.resize(rows * cols * 3);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            uint8_t *p = (uint8_t *)img.ptr(i) + j * 3;

            for (int k = 0; k < 3; k++) {
                result[k * rows * cols + i * cols + j] = (*(p + k) - mean) * scale;
            }
        }
    }

    return 0;
}

std::vector<std::vector<BoxItem>> yolov5_decoding(
    const std::vector<int> &raw_heights,         // 每张图片的高度
    const std::vector<int> &raw_widths,          // 每张图片的宽度
    const std::vector<std::vector<float>> &data, // YOLO可能有3+个输出Tensor, 这里依次存放每个输出
    const std::vector<std::vector<int>> &shape   // 这是对应每个输出的尺度, 元素格式: batch_size x anchors_each_grp x grid_size x grid_size x (5 + class_num)
) {
    std::vector<std::vector<BoxItem>> batch_results;

    // 数据校验
    if (data.size() != shape.size() || raw_heights.size() != raw_widths.size()) {
        std::cerr << "decoding error: data and shape size do not match" << std::endl;
        return batch_results;
    }

    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i][0] != (int)raw_heights.size()) {
            std::cerr << "decoding error: batch size does not match" << std::endl;
            return batch_results;
        }

        size_t len = 1;

        for (auto x: shape[i]) {
            len *= x;
        }

        if (len != data[i].size()) {
            std::cerr << "decoding error: invalid data: index " << i << ", shape count: " << len << ", data count: " << data[i].size() << std::endl;
            return batch_results;
        }
    }

    std::cout << "decoding info: data verified" << std::endl;

    for (int b = 0; b < int(raw_widths.size()); b++) {
        std::vector<BoxItem> cache;
        int raw_width = raw_widths[b];
        int raw_height = raw_heights[b];

        for (size_t layer = 0; layer < data.size(); layer++) {
            const float *out = &data[layer][0];
            int anchors_each_grp = shape[layer][1];
            int grid_size = shape[layer][2];

            for (int k = 0; k < anchors_each_grp; k++) {
                for (int i = 0; i < grid_size; i++) {
                    for (int j = 0; j < grid_size; j++) {
                        const float *grid_data = &out[
                            b * anchors_each_grp * grid_size * grid_size * (CLASS_NUM + 5) +
                            k * grid_size * grid_size * (CLASS_NUM + 5) +
                            i * grid_size * (CLASS_NUM + 5) +
                            j * (CLASS_NUM + 5)
                        ];
                        float max_score = 0.f;
                        int max_cls = -1;

                        for (int c = 0; c < CLASS_NUM; c++) {
                            float score = sigmoid(grid_data[5 + c]) * sigmoid(grid_data[4]);

                            if (score > CONF_THRESH && score > max_score) {
                                max_score = score;
                                max_cls = c;
                            }
                        }

                        if (max_cls >= 0) {
                            float cx = (sigmoid(grid_data[0]) * 2.f - 0.5f + j) * strides[layer];
                            float cy = (sigmoid(grid_data[1]) * 2.f - 0.5f + i) * strides[layer];
                            float w = powf(sigmoid(grid_data[2]) * 2.f, 2) * anchors[layer][k * 2];
                            float h = powf(sigmoid(grid_data[3]) * 2.f, 2) * anchors[layer][k * 2 + 1];

                            float minus_x = 0.f;
                            float minus_y = 0.f;
                            float multi = 1.f;

                            if (raw_width >= raw_height) {
                                minus_y = float(raw_width - raw_height) / raw_width * 0.5f;
                                multi = float(raw_width) / INPUT_SIZE;
                            }
                            else {
                                minus_x = float(raw_height - raw_width) / raw_height * 0.5f;
                                multi = float(raw_height) / INPUT_SIZE;
                            }

                            BoxItem item;
                            item.x = std::min(std::max(0, int((cx - w / 2 - INPUT_SIZE * minus_x) * multi)), raw_width - 1);
                            item.y = std::min(std::max(0, int((cy - h / 2 - INPUT_SIZE * minus_y) * multi)), raw_height - 1);
                            item.w = std::min(int(w * multi), raw_width);
                            item.h = std::min(int(h * multi), raw_height);
                            item.conf = max_score;
                            item.class_id = max_cls;
                            item.class_name = ""; // 暂未使用

                            if (item.w > 0 && item.h > 0) {
                                cache.push_back(item);
                            }
                        }
                    }
                }
            }
        }

        std::sort(cache.begin(), cache.end(), [](const BoxItem &a, const BoxItem &b) -> bool {
            return a.conf > b.conf;
        });

        size_t cnt = cache.size();

        if (cnt > MAX_OUTPUT_BBOX_COUNT) {
            cnt = MAX_OUTPUT_BBOX_COUNT;
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        std::vector<int> class_ids;

        for (size_t i = 0; i < cnt; i++) {
            cv::Rect r(cache[i].x, cache[i].y, cache[i].w, cache[i].h);
            boxes.push_back(r);
            confs.push_back(cache[i].conf);
            class_ids.push_back(cache[i].class_id);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH, indices);
        std::vector<BoxItem> result;

        for (auto idx: indices) {
            BoxItem item;
            item.x = boxes[idx].x;
            item.y = boxes[idx].y;
            item.w = boxes[idx].width;
            item.h = boxes[idx].height;
            item.conf = confs[idx];
            item.class_id = class_ids[idx];
            result.push_back(item);
        }

        batch_results.push_back(result);
    }

    return batch_results;
}
