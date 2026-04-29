#pragma once
#include <memory>
#include <cuda_runtime.h>
#include "NvInfer.h"

// 1. TensorRT 对象的智能指针删除器
struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) { delete obj; } // 注意：如果是极老的 TRT 7，这里要写 obj->destroy();
    }
};
// 2. CUDA 显存 (Device Memory) 的删除器
struct CudaDeleter{
    void operator()(void *Obj)const {
        if(Obj){
            cudaFree(Obj);
        }
    }
};



// 3. CUDA Stream 的删除器
struct CudaStreamDeleter {
    void operator()(cudaStream_t stream) const {
        if (stream) { cudaStreamDestroy(stream); }
    }
};

// --- 定义易用的类型别名 (Aliases) ---
template <typename T>
using TrtPtr = std::unique_ptr<T, TRTDeleter>;

template <typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter>;

// cudaStream_t 本身就是指针，所以需要稍微处理一下
using CudaStreamPtr = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CudaStreamDeleter>;