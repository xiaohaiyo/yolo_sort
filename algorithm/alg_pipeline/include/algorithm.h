#pragma once
#include "yolov8.h"
#include "data.h"
#include <librealsense2/rs.hpp>
#include <thread>
#include "FrameDispatcher.h"
#include "my_tracker.h"

class Algorithm {
public:
    Algorithm();
    ~Algorithm();
    bool start();
    bool stop();
    bool init(queque_mutex<input_alo>* input_queue, queque_mutex<output_alo> *output_queue, yolov8Config cfg);
    bool get_output();
    bool is_active() const;
    bool get_latest_output(output_alo &out,int &last_seq);
    bool updateDynamicParams(yolov8Config dynamic_cfg);
    yolov8Config get_latest_config();
private:
    void capture_loop();
    StreamGroup _streamGroup;
    std::unique_ptr<yolov8> _yolov8;
    std::unique_ptr<my_tracker> _sort;
    queque_mutex<input_alo>* _input_queue;
    queque_mutex<output_alo>* _output_queue;


    // 输入数据
    input_alo _input;
    int _inputid = 0;
    // 输出数据
    output_alo _output;
    //消费者线程
    std::thread _capture_thread;
    std::atomic<bool> _is_active;
};