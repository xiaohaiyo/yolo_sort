#pragma once
#include "tracker.h"
// #include"FeatureTensor.h"
#include "model.h"
// #include"FeatureTensor.h"
#include<memory>
#include<vector>
#include<string>
#include"yolo_datatype.h"
class my_tracker{
    public:
        my_tracker();
        ~my_tracker();
        bool init();
        std::vector<char> loadEngineFile(const std::string &path);
    private:
    std::unique_ptr<tracker> _tracker;
    // std::unique_ptr<FeatureTensor> _feature_tensor;  
    std::unique_ptr<DETECTIONS> _d1;
     std::unique_ptr<DETECTIONS> _d2;
    //输出信息
    int width;
    int height;
    

    public:
        void update(output_alo &input);
    

};

