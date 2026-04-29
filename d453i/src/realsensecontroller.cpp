#include "RealSenseController.h"
#include <iostream>

// 构造函数：初始化状态
RealSenseController::RealSenseController() : _is_streaming(false)
{
}

// 析构函数：对象销毁前检查是否关闭
RealSenseController::~RealSenseController()
{
    stop();
}

bool RealSenseController::init(queque_mutex<input_alo>* _input_queue)
{
    this->_input_queue = _input_queue;
    auto list = _ctx.query_devices();
    if (list.size() == 0) {
        std::cerr << "❌ 未检测到RealSense设备！" << std::endl;
        return false;

    }
    _active_device = list.front(); // 获取第一个连接的相机
    return true;
}
bool RealSenseController::start(StreamGroup &currentGroup1, StreamGroup &currentGroup2)
{
    try{
        _cfg.disable_all_streams();
        auto configure_stream = [this](const StreamGroup &group)
        {
            if (group.options.empty())
                return false;
            StreamOption opt = group.options[0];
            if (group.streamType == RS2_STREAM_INFRARED)
            {
                this->_cfg.enable_stream(group.streamType, 2, opt.width, opt.height, opt.format, opt.fps);
            }
            else
            {
                this->_cfg.enable_stream(group.streamType, -1, opt.width, opt.height, opt.format, opt.fps);
            }
        };
        _cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 60);
        configure_stream(currentGroup1);
        configure_stream(currentGroup2);
    this->_pipe.start(_cfg);

    this->_is_streaming = true;
    this->_worker_thread = std::thread(&RealSenseController::capture_loop, this);

    return true;
    }
    catch (const rs2::error &e)
{
    std::cerr << "❌ RealSense错误: " << e.what() << std::endl;
    return false;
}
}

// 停止摄像机逻辑
void RealSenseController::stop()
{
    if (_is_streaming)
    {
        try
        {
         _is_streaming = false;
          _pipe.stop();
        if (_worker_thread.joinable())
        {
                _worker_thread.join(); // 等待后台线程安全退出
        }
           
            
        std::cout << "RealSense 摄像机已安全关闭。" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "关闭摄像机时出错: " << e.what() << std::endl;
        }
    }
}

void RealSenseController::capture_loop()
{
    while (_is_streaming)
    {
        try
        {   
           
            rs2::frameset fs = _pipe.wait_for_frames(100); 
            

        //    rs2::align align_to_color(RS2_STREAM_COLOR);;
           
            // auto fs = align_to_color.process(fs1);

// 3. 获取各视频流帧
            auto color_f = fs.get_color_frame();
            auto depth_f = fs.get_depth_frame();
            // 注意：因为你在 start 里配置的是索引 1，所以这里对应取 index 1
            auto ir_f    = fs.get_infrared_frame(2); // 右红外

            if (!color_f || !depth_f || !ir_f) continue;

// --- 实时从相机硬件提取参数 ---
            static int frame_count = 0;

// if (frame_count++ % 100 == 0) 
// {
//     // 1. 提取 Profile
//     auto c_prof = color_f.get_profile().as<rs2::video_stream_profile>();
//     auto i_prof = ir_f.get_profile().as<rs2::video_stream_profile>();
    
//     // 2. 提取内参 (Intrinsics)
//     auto c_intrin = c_prof.get_intrinsics();
//     auto i_intrin = i_prof.get_intrinsics();

//     // 3. 提取外参 (Extrinsics: IR2 -> Color)
//     auto ir2_to_color = i_prof.get_extrinsics_to(c_prof);

//     // 4. 获取深度缩放
//     auto depth_sensor = _pipe.get_active_profile().get_device().first<rs2::depth_sensor>();
//     float scale = depth_sensor.get_depth_scale();

//     // --- 打印输出 ---
//     std::cout << std::fixed << std::setprecision(6);
//     std::cout << "\n>>>>>> 从硬件实时读取的相机全参数 [Frame: " << frame_count << "] <<<<<<" << std::endl;
    
//     std::cout << "--- 1. 彩色相机内参 (Color Intrinsics) ---" << std::endl;
//     std::cout << "const float COLOR_FX   = " << c_intrin.fx << "f;" << std::endl;
//     std::cout << "const float COLOR_FY   = " << c_intrin.fy << "f;" << std::endl;
//     std::cout << "const float COLOR_PPX  = " << c_intrin.ppx << "f;" << std::endl;
//     std::cout << "const float COLOR_PPY  = " << c_intrin.ppy << "f;" << std::endl;
//     std::cout << "// 畸变模型: " << rs2_distortion_to_string(c_intrin.model) << std::endl;
//     std::cout << "const float COLOR_D[5] = { " << c_intrin.coeffs[0] << "f, " << c_intrin.coeffs[1] << "f, " 
//               << c_intrin.coeffs[2] << "f, " << c_intrin.coeffs[3] << "f, " << c_intrin.coeffs[4] << "f };" << std::endl;

//     std::cout << "\n--- 2. 右红外相机内参 (IR2 Intrinsics) ---" << std::endl;
//     std::cout << "const float IR2_FX    = " << i_intrin.fx << "f;" << std::endl;
//     std::cout << "const float IR2_FY    = " << i_intrin.fy << "f;" << std::endl;
//     std::cout << "const float IR2_PPX   = " << i_intrin.ppx << "f;" << std::endl;
//     std::cout << "const float IR2_PPY   = " << i_intrin.ppy << "f;" << std::endl;
//     std::cout << "// 畸变模型: " << rs2_distortion_to_string(i_intrin.model) << std::endl;
//     std::cout << "const float IR2_D[5]  = { " << i_intrin.coeffs[0] << "f, " << i_intrin.coeffs[1] << "f, " 
//               << i_intrin.coeffs[2] << "f, " << i_intrin.coeffs[3] << "f, " << i_intrin.coeffs[4] << "f };" << std::endl;

//     std::cout << "\n--- 3. 空间外参 (IR2 -> Color) ---" << std::endl;
//     std::cout << "const float T_X = " << ir2_to_color.translation[0] << "f;" << std::endl;
//     std::cout << "const float T_Y = " << ir2_to_color.translation[1] << "f;" << std::endl;
//     std::cout << "const float T_Z = " << ir2_to_color.translation[2] << "f;" << std::endl;

//     std::cout << "\nconst float R[9] = {" << std::endl;
//     std::cout << "    " << ir2_to_color.rotation[0] << "f, " << ir2_to_color.rotation[1] << "f, " << ir2_to_color.rotation[2] << "f," << std::endl;
//     std::cout << "    " << ir2_to_color.rotation[3] << "f, " << ir2_to_color.rotation[4] << "f, " << ir2_to_color.rotation[5] << "f," << std::endl;
//     std::cout << "    " << ir2_to_color.rotation[6] << "f, " << ir2_to_color.rotation[7] << "f, " << ir2_to_color.rotation[8] << "f" << std::endl;
//     std::cout << "};" << std::endl;

//     std::cout << "\n--- 4. 其他常量 ---" << std::endl;
//     std::cout << "const float DEPTH_SCALE = " << scale << "f;" << std::endl;
//     std::cout << "const int DEPTH_W = " << i_intrin.width << ";" << std::endl;
//     std::cout << "const int DEPTH_H = " << i_intrin.height << ";" << std::endl;
//     std::cout << "========================================================\n" << std::endl;
// }

            if (fs)
            {
                // 2. 提取原生帧
                rs2::video_frame color_f = fs.get_color_frame();
                rs2::video_frame ir_f = fs.get_infrared_frame(2); // 左红外
                rs2::depth_frame depth_f = fs.get_depth_frame();

                if (!color_f || !ir_f || !depth_f) continue;

                // 3. 构造输出结构体
                input_alo input;
                // printf("color_f  %p\n", &color_f);
                // --- 处理彩色图 (RGB) ---
                input.color = cv::Mat(cv::Size(color_f.get_width(), color_f.get_height()), 
                                     CV_8UC3, (void *)color_f.get_data(), cv::Mat::AUTO_STEP).clone();

                // --- 处理红外图 (灰度转 BGR) ---
                cv::Mat ir_single = cv::Mat(cv::Size(ir_f.get_width(), ir_f.get_height()), 
                                           CV_8UC1, (void *)ir_f.get_data(), cv::Mat::AUTO_STEP);
                cv::cvtColor(ir_single, input.infrared, cv::COLOR_GRAY2BGR); // cvtColor 会自动完成内存拷贝，无需再 .clone()

                // --- 处理深度图 (16位) ---
                input.depth = cv::Mat(cv::Size(depth_f.get_width(), depth_f.get_height()), 
                                     CV_16UC1, (void *)depth_f.get_data(), cv::Mat::AUTO_STEP).clone();

                // 4. 推送到输入队列供算法线程使用
                if (this->_input_queue) {
                    // 使用 std::move 转移 Mat 的控制权，避免不必要的拷贝
                    this->_input_queue->push(std::move(input)); 
                }
            }
        }
        catch (const std::exception &e) {
            // 可以在这里打个日志，防止静默失败
            continue;
        }
    }
}
// 状态检查
bool RealSenseController::is_active() const
{
    return _is_streaming;
}


bool RealSenseController::get_supported_streams(std::vector<StreamGroup>& result)
{
    result.clear();
    std::map<rs2_stream, std::vector<StreamOption>> temp_map;

    for (auto&& sensor : _active_device.query_sensors()) {
        for (auto&& profile : sensor.get_stream_profiles()) {

            auto video = profile.as<rs2::video_stream_profile>();
            if (!video) continue;

            rs2_stream streamType = video.stream_type();
            if(video.fps()!=60) {continue;}
            if(video.width()!=640) {continue;}
            if( streamType==RS2_STREAM_DEPTH) {continue;}
            StreamOption opt;
            opt.width  = video.width();
            opt.height = video.height();
            opt.fps    = video.fps();
            opt.format = video.format();

            temp_map[streamType].push_back(opt);
        }
    }

    for (auto& [type, opts] : temp_map) {

        std::set<std::string> seen;
        std::vector<StreamOption> unique_opts;

        for (auto& o : opts) {
            std::string key = std::to_string(o.width) + "_" +
                              std::to_string(o.height) + "_" +
                              std::to_string(o.fps) + "_" +
                            rs2_format_to_string(o.format);  

            if (seen.insert(key).second) {
                unique_opts.push_back(o);
            }
        }

        StreamGroup group;
        group.streamType = type;
        group.options = unique_opts;

        result.push_back(group);
    }
    return true;

}