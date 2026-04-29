#pragma once
#include <librealsense2/rs.hpp>
#include <QString>
#include <QMetaType>
#include <vector>

struct StreamOption {
    int width;
    int height;
    int fps;
    rs2_format format;
};

// ⭐ 必须紧跟在类型后面
Q_DECLARE_METATYPE(StreamOption)

struct StreamGroup {
    rs2_stream streamType;              // Color / Depth / Infrared
    std::vector<StreamOption> options;
};

#ifndef CAMERA_CONFIG_H
#define CAMERA_CONFIG_H

#include <opencv2/opencv.hpp>

namespace CameraConfig {
    // --- 1. 彩色相机内参 (Color Intrinsics) ---
    const float COLOR_FX  = 604.723206f;
    const float COLOR_FY  = 604.791870f;
    const float COLOR_PPX = 330.451080f;
    const float COLOR_PPY = 243.509583f;
    // (彩色相机的 Inverse Brown Conrady 畸变系数全为 0)

    // --- 2. 右红外相机原始内参 (IR2 Intrinsics) ---
    const float IR2_FX    = 386.737488f;
    const float IR2_FY    = 386.737488f;
    const float IR2_PPX   = 326.854797f;
    const float IR2_PPY   = 235.215530f;

    // --- 3. 空间外参：右红外 (IR2) 到 彩色 (Color) ---
    // (注意：T_Y 和 T_Z 已更新为最新硬件校准数据)
    // const float T_X =  0.014956f; 
    const float T_X = 0.064951f;

    // const float T_Y = -0.000184f; // 更新
    const float T_Y = -0.001399f; // 更新
    // const float T_Z =  0.000306f; // 更新
    const float T_Z = 0.000176f; // 更新

    const float R[9] = {
        0.999701f, -0.024304f, -0.002611f,
        0.024291f,  0.999693f, -0.004792f,
        0.002726f,  0.004727f,  0.999985f
    };

    // --- 4. 深度与分辨率常量 ---
    const float DEPTH_SCALE = 0.001000f;
    const int DEPTH_W = 640; // 新增：显式记录标定分辨率
    const int DEPTH_H = 480; // 新增：显式记录标定分辨率

    // --- 5. 辅助函数：获取 OpenCV 矩阵格式 ---

    // 内参矩阵 K
    inline cv::Mat get_IR_K() {
        return (cv::Mat_<float>(3, 3) << 
            IR2_FX, 0,      IR2_PPX,
            0,      IR2_FY, IR2_PPY,
            0,      0,      1);
    }

    // 畸变系数矩阵 D
    // 硬件输出表明模型为 Brown Conrady 且全为 0，这代表硬件已预先完成去畸变
    inline cv::Mat get_IR_D() {
        return (cv::Mat_<float>(1, 5) << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}

#endif // CAMERA_CONFIG_H