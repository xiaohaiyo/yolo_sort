#ifndef REALSENSE_CONTROLLER_H
#define REALSENSE_CONTROLLER_H

#include <librealsense2/rs.hpp>
#include <mutex>
#include <thread>
#include <atomic>

#include "data.h"
#include "FrameDispatcher.h"
#include "algorithm.h"

class RealSenseController {
public:
    // 构造函数
    RealSenseController();

    // 析构函数（确保安全关闭设备）
    ~RealSenseController();

    // 启动摄像机
    bool start(StreamGroup& currentGroup1, StreamGroup& currentGroup2);

    // 停止采集并释放资源
    void stop();

    // 获取最新帧
    bool get_latest_frameset(rs2::frameset& outFs, int& imgid);
    bool get_supported_streams(std::vector<StreamGroup>& result);
    // 是否正在运行
    bool is_active() const;

    // 初始化分发队列
    bool init(queque_mutex<input_alo>* dispatcher);

private:
    // 采集线程函数
    void capture_loop();

private:
    std::thread _worker_thread;
    std::atomic<bool> _is_streaming{false};

    queque_mutex<input_alo>* _input_queue{nullptr};

    rs2::pipeline  _pipe;
    rs2::config    _cfg;
    rs2::colorizer _color_map;
    rs2::frameset  _current_frames;

    mutable std::mutex _data_mutex;

    // 设备信息
    rs2::device _active_device; 
    rs2::context _ctx;          
};

#endif // REALSENSE_CONTROLLER_H