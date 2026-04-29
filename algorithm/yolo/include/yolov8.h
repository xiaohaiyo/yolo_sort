#pragma once
#include "yolo_datatype.h"
#include "logg.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "trt_utils.h"
using namespace std;
class yolov8 {
    public:
        yolov8(yolov8Config config);
        ~yolov8();
        bool init();
        bool prescope(cv::Mat& color_img, cv::Mat& infrared_img);
        bool infer();
        bool postscope(output_alo &output);
        bool get_engine();
        vector<char> loadEngineFile(const string& path);
        bool updateDynamicParams(const yolov8Config& config);
        yolov8Config getConfig();
    private:
       
        //推理数据
        Logger logger;
        TrtPtr<nvinfer1::IRuntime> runtime=nullptr;
        TrtPtr<nvinfer1::ICudaEngine> engine=nullptr;
        TrtPtr<nvinfer1::IExecutionContext> context=nullptr;
        CudaStreamPtr stream=nullptr;
        size_t input_size;
        size_t output_size;
      
       
        //后处理数据
        CudaPtr<float>infer_d_output=nullptr;//GPU推理原输出
        std::vector<float> post_h_output; // CPU 端输出
        CudaPtr<float> post_d_output;// GPU端最终输出
        int *dections_sums=nullptr;
        
        //前处理
        CudaPtr<uint8_t>pre_d_input=nullptr;
        CudaPtr<float>pre_d_output=nullptr;
    

        //配置信息
         yolov8Config _config;
         std::mutex _config_mutex;


        
};