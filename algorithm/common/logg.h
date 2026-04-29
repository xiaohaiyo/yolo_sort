#pragma once
#include "NvInfer.h"
#include <iostream>
#include "cuda_runtime.h"
#include <cstdio>

#define errCudaCheck(ans)  _errCudaCheck((ans), __FILE__, __LINE__)

using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};


static inline cudaError _errCudaCheck(cudaError_t code, const char *file, int line)
{
   if(code != cudaSuccess) 
   {
        fprintf(stderr,"cuda错误:%s, 文件位置：%s：%d\n", cudaGetErrorString(code), file, line);
        // 这里建议保留 return，或者在严苛环境下直接调用 exit(1)
        return code;
   }
   return code;
}