#pragma once
#include <string>
#include "NvInfer.h"
#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
struct yolo_dection{
    float x, y, w, h, conf, cls,id;
};

struct yolo_output{
    std::vector<yolo_dection> yolooutput;
    std::vector<int> dections_sums;
};

struct input_alo{
    cv::Mat color;
    cv::Mat infrared;
    cv::Mat depth;
};
struct output_alo{
    std::vector<yolo_dection> sort_output;
    std::vector<yolo_dection> yolo_output;
    std::vector<int> dections_sums;
    input_alo imgs;

};

struct yolov8Config {


    // 置信度阈值：用于过滤检测到的物体
    // 只有概率大于此值的检测结果才会被保留（范围 0.0 ~ 1.0）
    float probabilityThreshold = 0.45f;

    // 非极大值抑制 (NMS) 阈值
    // 用于处理重叠的检测框。值越小，对重叠框的筛选越严格（去重效果更强）
    float nmsThreshold = 0.25f;

    // 最大检测目标数
    // 单张图像中允许返回的最大目标数量，防止检测结果过多导致性能下降
    int topK = 100;

    // 类别名称列表（默认使用 COCO 数据集的 80 类）
    std::vector<std::string> classNames = {
        "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",     "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",         "donut",         "cake",
        "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",         "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"};
    std::vector<int> targetClassId = {0};
    int target_width = 640;  // 输入图像宽度
    int target_height = 640; // 输入图像高度
    int or_width = 640; // 原始图像宽度
    int or_height = 480; // 原始图像高度
    int batchSize = 2; // 批处理大小
    std::string engineFileName = "/home/nvidia/fu_projext/orin_my_yolov8/input/yolov8n.engine"; // 引擎文件版本号（用于管理不同配置的引擎文件）
    std::string onnxFileName = "/home/nvidia/fu_projext/orin_my_yolov8/input/yolov8n.onnx";   // ONNX 模型文件路径
    
};

