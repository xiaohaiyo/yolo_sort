#include "yolov8.h"
#include <opencv2/opencv.hpp> // 包含 OpenCV 核心功能
#include <opencv2/dnn.hpp>
#include <fstream> // 包含文件流功能
#include "NvOnnxParser.h"
#include "trt_utils.h"
#include "yolov8cu.h"
using namespace nvonnxparser; // 包含 ONNX 模型解析功能
using namespace nvinfer1;     // 专门包含 DNN 模块的功能
yolov8::yolov8(yolov8Config config) : _config(config)
{
}

yolov8::~yolov8()
{
}

bool yolov8::init()
{
    if (this->_config.engineFileName == "")
    {
        std::cerr << "引擎文件为空从ONNX构建引擎文件!" << std::endl;
        if (get_engine())
        {
            std::cout << "成功构建引擎文件!" << std::endl;
        }
        else
        {
            std::cerr << "构建引擎文件失败!" << std::endl;
            return false;
        }
    }
    input_size = 3 * this->_config.target_width * this->_config.target_height * sizeof(float);
    output_size = sizeof(float);

    this->runtime.reset(createInferRuntime(this->logger)); // 创建 TensorRT 运行时
    if (this->runtime == nullptr)
    {
        std::cerr << "创建 TensorRT 运行时失败！" << std::endl;
        return false;
    }
    std::vector<char> modelData = loadEngineFile(this->_config.engineFileName);                   // 加载 .engine 文件到内存
    this->engine.reset(this->runtime->deserializeCudaEngine(modelData.data(), modelData.size())); // 反序列化生成的 Engine
    if (this->engine == nullptr)
    {
        std::cerr << "反序列化 Engine 失败！" << std::endl;
        return false;
    }
    this->context.reset(this->engine->createExecutionContext()); // 创建执行上下文
    if (this->context == nullptr)
    {
        std::cerr << "创建执行上下文失败！" << std::endl;
        return false;
    }
    const char *inputName = engine->getIOTensorName(0);
    const char *outputName = engine->getIOTensorName(1);
    nvinfer1::Dims4 input_shape(this->_config.batchSize, 3, this->_config.target_height, this->_config.target_width); // 设置输入张量的维度
    context->setInputShape(inputName, input_shape);                                                                   // 设置输入张量的形状
    auto output_dims = context->getTensorShape(outputName);                                                           // 获取输出张量的维度

    for (int i = 0; i < output_dims.nbDims; i++)
    {
        output_size *= output_dims.d[i];
    }
    cudaStream_t raw_stream = nullptr;
    cudaStreamCreate(&raw_stream);

    this->stream.reset(raw_stream); // 创建 CUDA

    // 前处理的输入
    uint8_t *d_in_ptr = nullptr;
    errCudaCheck(cudaMalloc(&d_in_ptr, 640 * 480 * 3 * this->_config.batchSize)); // 分配输入缓冲区
    this->pre_d_input.reset(d_in_ptr);

    // 前处理的输出，也是推理的输入
    float *d_out_ptr = nullptr;
    errCudaCheck(cudaMalloc(&d_out_ptr, input_size * this->_config.batchSize)); // 分配输入缓冲区
    this->pre_d_output.reset(d_out_ptr);

    // 推理原始输出，也是后处理的输入
    float *d_out_ptr1 = nullptr;
    errCudaCheck(cudaMalloc(&d_out_ptr1, output_size));
    this->infer_d_output.reset(d_out_ptr1);

    context->setTensorAddress(inputName, pre_d_output.get());
    context->setTensorAddress(outputName, infer_d_output.get());

    // 后处理的输出
    float *post_out_ptr = nullptr;
    errCudaCheck(cudaMalloc(&post_out_ptr, (this->_config.batchSize * (1 + 7 * this->_config.topK)) * sizeof(float)));
    this->post_d_output.reset(post_out_ptr);

    this->post_h_output.resize(this->_config.batchSize * (1 + this->_config.topK * 7));

    errCudaCheck(cudaMalloc(&this->dections_sums, this->_config.batchSize * sizeof(int)));
    errCudaCheck(cudaMemset(this->dections_sums, 0, this->_config.batchSize * sizeof(int)));

    return true;
}

std::vector<char> yolov8::loadEngineFile(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

bool yolov8::get_engine()
{

    IBuilder *builder = createInferBuilder(this->logger); // 创建 TensorRT 构建器
    if (!builder)
        return false;

    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(flag);
    if (!network)
        return false;

    IParser *parser = createParser(*network, logger);
    if (!parser)
        return false;

    // 解析 ONNX 模型文件
    if (!parser->parseFromFile(this->_config.onnxFileName.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING)))
    {
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cerr << "解析错误: " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    IBuilderConfig *config = builder->createBuilderConfig();

    // 修复位移错误：1ULL << 30 表示 1GB 内存 (1024 * 1024 * 1024 字节)
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 20);

    // 构建序列化的网络模型
    IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (!serializedModel)
    {
        std::cerr << "构建序列化网络失败！" << std::endl;
        return false;
    }

    // ==========================================
    // 2. 将模型保存到本地生成 .engine 文件
    // ==========================================
    std::ofstream engine_file(this->_config.engineFileName, std::ios::binary); // 以二进制模式打开

    if (!engine_file)
    {
        std::cerr << "无法打开文件以保存 Engine！路径: " << this->_config.engineFileName << std::endl;
    }
    else
    {
        // 将内存中的数据写入文件
        engine_file.write(static_cast<const char *>(serializedModel->data()), serializedModel->size());
        engine_file.close();
        std::cout << "成功导出 Engine 文件，保存至: " << this->_config.engineFileName << std::endl;
    }
    delete parser;
    delete network;
    delete config;
    delete builder;
    delete serializedModel;
    return true; // 修复：添加返回值
}

bool yolov8::prescope(cv::Mat &color_img, cv::Mat &infrared_img)
{
    if (color_img.empty() || infrared_img.empty())
    {
        std::cerr << "Error: Input images are empty!" << std::endl;
        return false;
    }
    // std::cout << "开始前处理" << std::endl;
    preprocess_cuda(color_img, infrared_img, this->pre_d_input.get(), this->pre_d_output.get(), this->_config.target_width, this->_config.target_height, color_img.cols, color_img.rows);
    // std::cout << "前处理完成" << std::endl;

    // cv::dnn::blobFromImage(color_img,
    //                        this->blob_color,
    //                        1.0 / 255.0,         // 归一化 (0-255 -> 0-1)
    //                        cv::Size(_config.width, _config.height),  // 调整尺寸
    //                        cv::Scalar(0, 0, 0), // 均值减法 (Mean Subtraction)
    //                        true,                // 交换 RB (如果是 BGR 转 RGB 就选 true)
    //                        false);
    // cv::dnn::blobFromImage(infrared_img,
    //                        this->blob_infrared,
    //                        1.0 / 255.0,         // 归一化 (0-255 -> 0-1)
    //                        cv::Size(_config.width, _config.height),  // 调整尺寸
    //                        cv::Scalar(0, 0, 0), // 均值减法 (Mean Subtraction)
    //                        false,               // 交换 RB (如果是 BGR 转 RGB 就选 true)
    //                        false);

    return true;
}
bool yolov8::infer()

{
    // std::cout << "开始推理" << std::endl;
    context->enqueueV3(this->stream.get());
    cudaStreamSynchronize(this->stream.get());
    // std::cout << "推理完成" << std::endl;
    return true;
}

bool yolov8::postscope(output_alo &output)
{
    // std::cout << "开始后处理" << std::endl;
    postprocess_cuda(this->infer_d_output.get(), this->post_d_output.get(), this->post_h_output.data(), this->_config, output);
    // 打印 output
    // std::cout << "--- YOLO 推理结果 ---" << std::endl;

    // 1. 打印检测到的目标总数
    // 根据你的结构体，总数存储在 detections_sums 这个 vector 中
    // if (!this->_output.dections_sums.empty())
    // {
    //     std::cout << "检测到的目标总数 (num_boxes): " << this->_output.dections_sums[0] << std::endl;
    // }

    // // 2. 遍历检测结果 vector
    // // 使用 yolooutput.size() 动态获取实际检测到的框数量
    // std::cout << "检测框详情 (Boxes):" << std::endl;

    // for (size_t i = 0; i < this->_output.yolooutput.size(); i++)
    // {
    //     // 这里的 item 就是单个 yolo_dection 对象
    //     const auto &item = this->_output.yolooutput[i];

    //     std::cout << "  [目标 " << i << "]:" << std::endl;
    //     std::cout << "    类别 ID: " << item.cls << std::endl;
    //     std::cout << "    置信度: " << item.<< std::endl;
    //     std::cout << "    位置: [x:" << item.x << ", y:" << item.y
    //               << ", w:" << item.w << ", h:" << item.h << "]" << std::endl;
    // }

    // if (this->_output.yolooutput.empty())
    // {
    //     std::cout << "  (未检测到任何目标)" << std::endl;
    // }
    // std::cout << "后处理完成" << std::endl;
    return true;
}

bool yolov8::updateDynamicParams(const yolov8Config &config)
{
    if (config.probabilityThreshold < 0.0f || config.probabilityThreshold > 1.0f)
    {
        return false;
    }
    std::unique_lock<std::mutex> lock(this->_config_mutex, std::try_to_lock);
    if (lock.owns_lock())
    {
        this->_config.probabilityThreshold = config.probabilityThreshold;
        this->_config.nmsThreshold = config.nmsThreshold;
        this->_config.topK = config.topK;
        this->_config.targetClassId = config.targetClassId;
         return true; // 更新成功返回 true
        return true; // 更新成功返回 true
    }
    return false;
}


yolov8Config yolov8::getConfig()
{
    return this->_config;
}