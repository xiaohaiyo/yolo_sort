#include "my_tracker.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "yolo_datatype.h"
#include "data.h"
my_tracker::my_tracker() {}
my_tracker::~my_tracker() {}

bool my_tracker::init() {
    try {
        // 1. 初始化追踪器逻辑 (参数通常为: max_cosine_distance, nn_budget)
        this->_tracker = std::make_unique<tracker>(0.2f, 100);
        
        // 2. 初始化特征提取引擎
        // this->_feature_tensor = std::make_unique<FeatureTensor>();
        // // 注意：原代码逻辑中 init() 返回 true 代表失败，已修正为符合直觉的判断
        // if (!this->_feature_tensor->init()) { 
        //     std::cerr << "FeatureTensor Engine Init Failed!" << std::endl;
        //     return false;
        // }

        // 3. 初始化检测容器缓存
        this->_d1 = std::make_unique<DETECTIONS>();
         this->_d2 = std::make_unique<DETECTIONS>();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "my_tracker init exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "my_tracker init unknown error!" << std::endl;
        return false;
    }
}

void my_tracker::update(output_alo &input) {
    if (input.yolo_output.empty()) {
        return;
    }

    this->_d1->clear(); 
    this->_d2->clear(); 

    const cv::Mat& depth = input.imgs.depth;
    int w = depth.cols;
    int h = depth.rows;

    int color_count = input.dections_sums.empty() 
                      ? (int)input.yolo_output.size() 
                      : input.dections_sums[0];

for (int i = 0; i < (int)input.yolo_output.size(); ++i) {
        auto det = input.yolo_output[i];
        
        // 初始值：默认使用红外原始坐标和大小
        float aligned_x = det.x;
        float aligned_y = det.y;
        float aligned_w = det.w;
        float aligned_h = det.h;

        // 仅对红外路进行处理
        if (i >= color_count) {
            // 1. 中值深度提取 (保持你现有的逻辑)
            int x0 = (int)(det.x - det.w * 0.5f);
            int y0 = (int)(det.y - det.h * 0.5f);
            int x1 = (int)(det.x + det.w * 0.5f);
            int y1 = (int)(det.y + det.h * 0.5f);

            std::vector<float> vals;
            vals.reserve(128);
            int step = std::max(1, (x1 - x0) / 20);
            int valid_count = 0;

            for (int yy = y0; yy <= y1; yy += step) {
                for (int xx = x0; xx <= x1; xx += step) {
                    if (xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
                    valid_count++;
                    unsigned short d = depth.at<unsigned short>(yy, xx);
                    if (d == 0) continue;
                    float depth_m = d * CameraConfig::DEPTH_SCALE;
                    if (depth_m > 0.1f && depth_m < 10.0f) vals.push_back(depth_m);
                }
            }

            float Z = -1.0f;
            if (!vals.empty() && valid_count >= 10) {
                size_t mid = vals.size() / 2;
                std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
                Z = vals[mid];
            }

            // 2. 精确投影变换 + 动态缩放
            if (Z > 0.1f) {
                // --- A. 去畸变 ---
                std::vector<cv::Point2f> src_pts = { cv::Point2f(det.x, det.y) };
                std::vector<cv::Point2f> dst_pts;
                cv::undistortPoints(src_pts, dst_pts, CameraConfig::get_IR_K(), CameraConfig::get_IR_D(), cv::noArray(), CameraConfig::get_IR_K());
                float ux = dst_pts[0].x;
                float uy = dst_pts[0].y;

                // --- B. IR 2D -> 3D ---
                float X_ir = (ux - CameraConfig::IR2_PPX) * Z / CameraConfig::IR2_FX;
                float Y_ir = (uy - CameraConfig::IR2_PPY) * Z / CameraConfig::IR2_FY;

                // --- C. IR 3D -> Color 3D (应用 R, T) ---
                float Xc = CameraConfig::R[0]*X_ir + CameraConfig::R[1]*Y_ir + CameraConfig::R[2]*Z + CameraConfig::T_X;
                float Yc = CameraConfig::R[3]*X_ir + CameraConfig::R[4]*Y_ir + CameraConfig::R[5]*Z + CameraConfig::T_Y;
                float Zc = CameraConfig::R[6]*X_ir + CameraConfig::R[7]*Y_ir + CameraConfig::R[8]*Z + CameraConfig::T_Z;

                if (Zc > 0.001f) {
                    // --- D. 映射回彩色中心点 ---
                    aligned_x = (Xc * CameraConfig::COLOR_FX / Zc) + CameraConfig::COLOR_PPX;
                    aligned_y = (Yc * CameraConfig::COLOR_FY / Zc) + CameraConfig::COLOR_PPY;

                    // --- E. 动态计算缩放因子 (s_factor) ---
                    // 这里的计算比死写 1.56 更精确，因为它考虑了 Zc 的变化
                    float s_factor = (CameraConfig::COLOR_FX / CameraConfig::IR2_FX) * (Z / Zc);
                    aligned_w = det.w * s_factor;
                    aligned_h = det.h * s_factor;
                }
            } else {
                // 如果拿不到深度，可以给一个保守的默认缩放 1.56
                // 这样即使深度丢失，框的位置可能会飘，但大小至少是对的
                aligned_w = det.w * 1.56f;
                aligned_h = det.h * 1.56f;
            }
        }
      
        // 3. 封装到检测行 (核心：基于新中心和新宽高重新计算左上角)
        DETECTION_ROW row;
        float tl_x = aligned_x - aligned_w * 0.5f;
        float tl_y = aligned_y - aligned_h * 0.5f;

        row.tlwh = DETECTBOX(tl_x, tl_y, aligned_w, aligned_h);
        row.confidence = det.conf;
        row.cls = det.cls;
        row.conf = det.conf;
        row.feature = FEATURESS::Zero(1, 512);
        row.feature(0, 0) = 1.0f;
        // row.sum = det.sum;
        if(i>=color_count) {
            this->_d2->push_back(row);
        }else{
            this->_d1->push_back(row);
        }
        
    }

    // --- 步骤 2 & 3：追踪器运行与结果回写 (逻辑同前) ---
    this->_tracker->predict();  
    if(this->_tracker->is_first&&this->_d1->size()>0) {
        //  this->_tracker->predict();  
        // this->_tracker->is_first = false;
        this->_tracker->update(*this->_d1,1); 
        this->_tracker->is_first = false;
    } else {
        if(this->_d1->size()>0){
            // printf("update 1\n");
        this->_tracker->update(*this->_d1,1);
        }
        if(this->_d2->size()>0){
            // printf("update 2\n");
        this->_tracker->update(*this->_d2,2);
        }
         
    }
    this->_tracker->update2();
    input.sort_output.clear();
    int confirmed_count = 0;
    for (auto &track : this->_tracker->tracks) {
        if (track.is_confirmed() && track.time_since_update <= 1) {
            DETECTBOX box = track.to_tlwh();
            yolo_dection out_det;
            out_det.x = box(0) + box(2) / 2.0f;
            out_det.y = box(1) + box(3) / 2.0f;
            out_det.w = box(2);
            out_det.h = box(3);
            out_det.id = (float)track.track_id; 
            out_det.cls = track.cls;   
            out_det.conf = track.conf; 
            input.sort_output.push_back(out_det);
            confirmed_count++;
        }
    }
    // input.dections_sums.clear();
    input.dections_sums[2] = confirmed_count;
}
std::vector<char> my_tracker::loadEngineFile(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return {};
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}