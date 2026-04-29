#include "algorithm.h"
#include "my_tracker.h"
Algorithm :: Algorithm():
_is_active(false),
_input_queue(nullptr),
_output_queue(nullptr),
_sort(nullptr)
{
   
};
Algorithm::~Algorithm()
{
    if (_is_active)
    {
        stop();
    }
}
bool Algorithm::init(queque_mutex<input_alo> *input_queue, queque_mutex<output_alo> *output_queue, yolov8Config cfg)
{
    try
    {
        _yolov8 = std::make_unique<yolov8>(cfg);
        // 1. 安全检查：防止 MainWindow 传了一个空指针进来
        if (!input_queue)
        {
            std::cerr << "初始化失败：传入的 input_queue 是空的！" << std::endl;
            return false;
        }
        if (!output_queue)
        {
            std::cerr << "初始化失败：传入的 output_queue 是空的！" << std::endl;
            return false;
        }
        
        // 2. 建立连接：算法类正式持有相机的队列地址
        this->_input_queue = input_queue;
        this->_output_queue = output_queue;
        
        // 3. 核心初始化：启动模型
        if (!_yolov8->init())
        {
            std::cerr << "初始化yolov8模型失败：" << std::endl;
            return false;
        }
        // 4. 初始化tracker
        _sort = std::make_unique<my_tracker>();
        if(!_sort->init()){
            std::cerr << "初始化sort失败：" << std::endl;
            return false;
        }
          


        return true; // 🌟 必须手动返回 true，表示一切正常
    }
    catch (const std::exception &e)
    { // 建议用 std::exception 捕获范围更广
        std::cerr << "初始化算法遇到异常：" << e.what() << std::endl;
        return false;
    }
}
void Algorithm::capture_loop()
{
    while (_is_active)
    {
        
        if (_input_queue && _input_queue->pop(this->_input,this->_inputid))
        {
            // 红外，可见两路检测
            if (!this->_input.color.empty() && !this->_input.infrared.empty())
            {
                // std::cout << "Color Frame: " << color.get_width() << "x" << color.get_height() << " | format: " << color.get_profile().format() << std::endl;
                // std::cout << "Infrared Frame: " << infrared.get_width() << "x" << infrared.get_height() << " | format: " << infrared.get_profile().format() << std::endl;

                // 1. 检测
                _output.dections_sums.resize(3, 0); // 确保 dections_sums 有足够的空间存储三个元素    
                _yolov8->prescope(this->_input.color, this->_input.infrared);
                _yolov8->infer();
                _yolov8->postscope(this->_output);
                // 2. 追踪
                this->_output.imgs = this->_input;
                printf("进入");
                _sort->update(this->_output);\
                printf("sort update\n");
                if (_output_queue) {
                    this->_output_queue->push(std::move(this->_output)); 
                }
            }
            else
            {
                std::cerr << "错误：缺少color和红外的图像数据" << std::endl;
            }
        }
    }
}
bool Algorithm::start()
{
    this->_is_active = true;
    this->_capture_thread = std::thread(&Algorithm::capture_loop, this);
    std::cout << "Algorithm 线程已启动" << std::endl;
    return true;
    
}

bool Algorithm::stop()
{
    if(this->_is_active){
        try{
            this->_is_active = false;
            this->_inputid = 0;
            if(_capture_thread.joinable()){
                _capture_thread.join();
                 std::cout << "Algorithm 线程已停止" << std::endl;
                return true;
               
            }
        }

   catch(const std::exception &e){
        std::cerr << "停止算法线程时出错：" << e.what() << std::endl;
        return false;
    }
} }

bool Algorithm::is_active() const
{
    return _is_active;
}


bool Algorithm::get_latest_output(output_alo &out,int &last_seq){
    if(_output_queue && _output_queue->pop(out,last_seq)){
        return true;
    }
    return false;
}
yolov8Config Algorithm::get_latest_config(){
    return _yolov8->getConfig();
}
bool Algorithm::updateDynamicParams(yolov8Config dynamic_cfg){
    return _yolov8->updateDynamicParams(dynamic_cfg);
}
