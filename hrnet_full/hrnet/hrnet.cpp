#include "hrnet.h"

#include <stdint.h>

static const int SCALE_STD = 200; // 尺度缩放标准值

// 根据自己情况, 提前确定的归一化数值
static const float means[3] = { 103.53, 116.28, 123.675 };
static const float stds[3] = { 57.375, 57.12, 58.395 };

// 计算 box 的中心坐标, 以及缩放尺度
void box_to_center_scale(
    std::vector<int> &box, // 格式: { x1, y1, x2, y2 }, 表示大图上一个人的矩形框
    int target_width, 
    int target_height, 
    std::vector<float> &center, // 存放 box 中心点
    std::vector<float> &scale
) {
	float box_width = box[2] - box[0];
	float box_height = box[3] - box[1];

	center[0] = box[0] + box_width * 0.5;
	center[1] = box[1] + box_height * 0.5;

	float aspect_ratio = (float)target_width / target_height; // 宽高比

	if (box_width > aspect_ratio * box_height) { // 太宽的情况
		box_height = box_width * 1.0 / aspect_ratio; // 重新调整高度
	}
	else if (box_width < aspect_ratio * box_height) { // 太窄的情况
		box_width = box_height * aspect_ratio; // 重新调整宽度
	}

	scale[0] = box_width / SCALE_STD; // 调整后与标准值的比例，作为缩放比例
	scale[1] = box_height / SCALE_STD;

	if (center[0] != -1) {
		scale[0] = scale[0] * 1.25; // 调整系数
		scale[1] = scale[1] * 1.25;
	}
}

// 归一化
void norm_img(const cv::Mat &img, std::vector<float> &data) {
    int rows = img.rows;
    int cols = img.cols;
    data.resize(img.channels() * rows * cols);

	for (int i = 0; i < rows; i++) {
		const uint8_t *p = img.ptr<uint8_t>(i);
        size_t offset = 0;

		for (int j = 0; j < cols; j++) {
            for (int k = 0; k < 3; k++) {
			    data[k * rows * cols + i * cols + j] = (*(p + offset++) - means[0]) / stds[0];
            }
		}
	}
}

// 符号函数
inline int sign(float x) {
	int w = 0;

	if (x > 0) {
		w = 1;
	}
	else if (x == 0) {
		w = 0;
	}
	else {
		w = -1;
	}

	return w;
}

// 对坐标进行旋转
std::vector<float> rotate_point(float x, float y, float rot_rad) {
	float sn = sin(rot_rad);
	float cs = cos(rot_rad);
	std::vector<float> new_pt { 0.0, 0.0 };
	new_pt[0] = x * cs - y * sn;
	new_pt[1] = x * sn + y * cs;
	return new_pt;
}

// 计算特定逻辑的点坐标
std::vector<float> get_3rd_point(const std::vector<float> &a, std::vector<float> &b) {
	std::vector<float> direct { a[0] - b[0], a[1] - b[1] };
	return std::vector<float> { b[0] - direct[1], b[1] + direct[0] };
}

// 计算仿射矩阵
cv::Mat get_affine_transform(
    const std::vector<float> &center, 
    const std::vector<float> &scale, 
    float rot, 
    std::vector<int> &img_size, 
    int inv
) {
	float src_w = scale[0] * SCALE_STD;
	int dst_w = img_size[0];
	int dst_h = img_size[1];
	float rot_rad = rot * 3.1415926535 / 180;

	std::vector<float> src_pt = rotate_point(0, -0.5 * src_w, rot_rad);
	std::vector<float> dst_pt{ 0.0, float(-0.5) * dst_w };

	std::vector<float> src1{ center[0] + src_pt[0], center[1] + src_pt[1] };
	std::vector<float> src2 = get_3rd_point(center, src1);

	std::vector<float> dst0{ float(dst_w * 0.5),float(dst_h * 0.5) };
	std::vector<float> dst1{ float(dst_w * 0.5) + dst_pt[0],float(dst_h * 0.5) + dst_pt[1] };
	std::vector<float> dst2 = get_3rd_point(dst0, dst1);

	if (inv == 0) {
		float a[6][6] = { { center[0], center[1], 1, 0, 0, 0 },
						  { 0, 0, 0, center[0], center[1], 1 },
						  { src1[0], src1[1], 1, 0, 0, 0 },
						  { 0, 0, 0, src1[0], src1[1], 1 },
						  { src2[0], src2[1], 1, 0, 0, 0 },
						  { 0, 0, 0, src2[0], src2[1], 1 } };
		float b[6] = { dst0[0], dst0[1], dst1[0], dst1[1], dst2[0], dst2[1] };
		cv::Mat a_1 = cv::Mat(6, 6, CV_32F, a);
		cv::Mat b_1 = cv::Mat(6, 1, CV_32F, b);
		cv::Mat result;
		solve(a_1, b_1, result, 0);
		cv::Mat dst = result.reshape(0, 2);
		return dst;
	}
	else {
		float a[6][6] = { { dst0[0], dst0[1], 1, 0, 0, 0 },
						  { 0, 0, 0, dst0[0], dst0[1], 1 },
						  { dst1[0], dst1[1], 1, 0, 0, 0 },
						  { 0, 0, 0, dst1[0], dst1[1], 1 },
						  { dst2[0], dst2[1], 1, 0, 0, 0 },
						  { 0, 0, 0, dst2[0], dst2[1], 1 } };
		float b[6] = { center[0], center[1], src1[0], src1[1], src2[0], src2[1] };
		cv::Mat a_1 = cv::Mat(6, 6, CV_32F, a);
		cv::Mat b_1 = cv::Mat(6, 1, CV_32F, b);
		cv::Mat result;
		solve(a_1, b_1, result, 0);
		cv::Mat dst = result.reshape(0, 2);
		return dst;
	}
}

// 单图的归一化
int get_norm_data(
    const cv::Mat &img, 
    std::vector<float> &result, 
    std::vector<float> &center, 
    std::vector<float> &scale
) {
    if (img.empty()) {
        return -1;
    }

    center.push_back(0);
    center.push_back(0);
    scale.push_back(0);
    scale.push_back(0);

    std::vector<int> box_max { 0, 0, img.cols - 1, img.rows - 1 }; // x1, y1, x2, y2
    std::vector<int> img_size { INPUT_WIDTH, INPUT_HEIGHT };

    box_to_center_scale(box_max, img_size[0], img_size[1], center, scale);

    cv::Mat input;
    cv::Mat tran = get_affine_transform(center, scale, 0, img_size, 0);

    cv::warpAffine(img, input, tran, cv::Size(img_size[0], img_size[1]), cv::INTER_LINEAR);
    norm_img(input, result);

    return 0;
}

// 解码（batch）
int hrnet_decoding(
    const std::vector<float> &heatmap, // 模型推理输出的数据, 只有一层 vector, 因为输入、输出都只有 1 路
    const std::vector<int> &shapes, // 模型推理输出的 shape
    const std::vector<std::vector<float> > &centers, // 每个图的中心点
    const std::vector<std::vector<float> > &scales,  // 每个图的缩放尺度
    std::vector<std::vector<cv::Point> > &keypoints // 存放结果
) {
    if (
        (int)shapes.size() != 4 || 
        (int)heatmap.size() != shapes[0] * shapes[1] * shapes[2] * shapes[3] || 
        (int)centers.size() != shapes[0] || 
        (int)scales.size() != shapes[0]
    ) {
        std::cerr << "ERROR: invalid shapes" << std::endl;
        return -1;
    }

    int batch_size = shapes[0];
    int num_joints = shapes[1];
	int heatmap_height = shapes[2];
	int heatmap_width = shapes[3];

    std::vector<int> img_size { heatmap_width, heatmap_height };
    std::vector<std::vector<cv::Point2f> > preds;

	for (int i = 0; i < batch_size; i++) {
        std::vector<cv::Point2f> pts;

        for (int j = 0; j < num_joints; j++) {
            pts.push_back(cv::Point(0.f, 0.f));
        }

        preds.push_back(pts);

        // 找最大响应, 确定坐标
		for (int j = 0; j < num_joints; j++) {
            int base = i * num_joints * heatmap_height * heatmap_width + j * heatmap_height * heatmap_width;
			float max_val = heatmap[base];
			int max_id = 0;

			for (int k = 1; k < heatmap_height * heatmap_width; k++) {
				int pos = base + k;

				if (heatmap[pos] > max_val) {
					max_val = heatmap[pos];
					max_id = k;
				}
			}

            int x_id = max_id % heatmap_width;
            int y_id = max_id / heatmap_width;

            if (max_val > 0) {
                preds[i][j].x = x_id;
                preds[i][j].y = y_id;
            }
            else {
                preds[i][j].x = 0.f;
                preds[i][j].y = 0.f;
            }
		}

        // 后处理
        for (int j = 0; j < num_joints; j++) {
            int x = int(preds[i][j].x);
            int y = int(preds[i][j].y);
            int base = i * num_joints * heatmap_height * heatmap_width + j * heatmap_height * heatmap_width;

            if (x > 1 && x < heatmap_width - 2 && y > 1 && y < heatmap_height - 2) {
                float x_plus_1 = heatmap[base + y * heatmap_width + x + 1];
                float x_minus_1 = heatmap[base + y * heatmap_width + x - 1];
                float y_plus_1 = heatmap[base + (y + 1) * heatmap_width + x];
                float y_minus_1 = heatmap[base + (y - 1) * heatmap_width + x];
                float diff_x = sign(x_plus_1 - x_minus_1) * 0.25;
                float diff_y = sign(y_plus_1 - y_minus_1) * 0.25;
                preds[i][j].x += diff_x;
                preds[i][j].y += diff_y;
            }
        }
        
        // 根据预处理时记录的中心点、缩放尺寸, 进行仿射变换
        cv::Mat M = get_affine_transform(centers[i], scales[i], 0, img_size, 1);

        for (int j = 0; j < num_joints; j++) {
            float vals[3] = { preds[i][j].x, preds[i][j].y, 1.f }; // ! 不是cv::Scalar
            cv::Mat pt(3, 1, M.type(), vals);
            cv::Mat new_pt = M * pt;
            preds[i][j].x = new_pt.at<float>(0, 0);
            preds[i][j].y = new_pt.at<float>(1, 0);
        }
	}

    keypoints.clear();

    for (size_t i = 0; i < preds.size(); i++) {
        std::vector<cv::Point> pts;

        for (int j = 0; j < num_joints; j++) {
            pts.push_back(cv::Point(int(preds[i][j].x), int(preds[i][j].y)));
        }

        keypoints.push_back(std::move(pts));
    }

    return 0;
}
