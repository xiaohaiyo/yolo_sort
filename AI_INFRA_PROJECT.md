# AI Infra 项目介绍：多源融合实时目标检测与跟踪系统

## 项目定位

本项目是一个面向边缘端的 **AI 基础设施（AI Infra）** 实战项目，聚焦于将深度学习目标检测模型从"能跑"推向"能用"——即在资源受限的嵌入式 GPU 平台上，实现多传感器融合、低延迟推理、实时跟踪的完整工程闭环。

## AI Infra 核心技术栈

### 1. 推理引擎层（Inference Engine）

| 组件 | 技术选型 | 说明 |
|---|---|---|
| 模型格式 | ONNX → TensorRT Engine | 模型从训练框架导出后，通过 TensorRT 编译优化为目标平台的高性能引擎 |
| 推理框架 | NVIDIA TensorRT (NvInfer) | 支持 layer fusion、precision calibration (FP16/INT8)、dynamic batching |
| 模型解析 | NvOnnxParser | ONNX 模型到 TensorRT 网络的自动转换 |

**工程要点**：
- 采用 batch_size=2 的批量推理，将可见光和红外两路图像合并为一个 batch 送入 GPU，单次 kernel launch 完成双路推理，最大化 GPU 利用率
- TensorRT engine 绑定固定的输入输出 tensor shape，避免运行时动态分配

### 2. GPU 计算层（CUDA Compute）

本项目将传统 CPU 上的图像预处理和后处理全部下沉到 CUDA kernel，实现端到端 GPU 流水线：

```
Camera Frame (CPU)
    │
    ▼ DMA / cudaMemcpy
GPU Memory
    │
    ├── prescope kernel:  bilinear resize (640×480 → 640×640) + normalize + HWC→CHW
    │
    ├── TensorRT infer:   YOLOv8n batch-2 forward pass
    │
    ├── postscope kernel1: confidence filter + class assignment
    │
    └── postscope kernel2: GPU-based NMS (Non-Maximum Suppression)
    │
    ▼ cudaMemcpy
CPU Results
```

**关键 CUDA kernel**：
- **预处理 kernel**：双线性插值缩放 + 像素归一化 + HWC→CHW 布局转换，一步完成
- **后处理 kernel 1**：逐 anchor 置信度过滤，输出有效检测框
- **后处理 kernel 2**：基于 IOU 矩阵的 GPU NMS，避免 CPU 端逐框比较的性能瓶颈

### 3. 多线程流水线层（Pipeline Architecture）

系统采用经典的 **生产者-消费者** 三线程架构：

```
Thread 1: Camera Producer    →  线程安全单槽队列  →  Thread 2: Algorithm Consumer
                                                    →  线程安全单槽队列  →  Thread 3: UI Consumer
```

- **单槽队列（Single-Slot Queue）**：对于实时视频流，只保留最新帧，丢弃旧帧。比 ring buffer 更简单，避免内存浪费
- **互斥锁保护**：`queque_mutex<T>` 使用 `std::mutex` + `std::condition_variable` 实现 push 覆盖、pop 阻塞
- **UI 定时刷新**：QTimer 每 15ms 触发一次，从输出队列取最新结果渲染，保证 UI 不阻塞算法线程

### 4. 多传感器融合层（Sensor Fusion）

这是本项目区别于普通 YOLO demo 的核心差异点：

**问题**：红外相机和可见光相机有不同的内参（焦距、主点）和坐标系，红外检测框无法直接叠加到可见光画面上。

**解决方案**：基于相机标定参数的几何投影对齐

```
红外检测框 (2D)
    │
    ▼ 去畸变 (cv::undistortPoints)
归一化坐标
    │
    ▼ 反投影 (红外内参矩阵逆)
红外相机空间 3D 点 (利用深度图获取 Z)
    │
    ▼ 刚体变换 (外参 R, T)
可见光相机空间 3D 点
    │
    ▼ 投影 (可见光内参矩阵)
可见光图像 2D 坐标
```

**工程价值**：这不仅仅是"画框"，而是实现了跨模态检测结果在同一坐标系下的统一表达，为后续的多目标跟踪提供了正确的输入。

### 5. 多目标跟踪层（Multi-Object Tracking）

采用 DeepSORT 算法，核心组件：

- **卡尔曼滤波器（Kalman Filter）**：预测目标在下一帧的状态（位置 + 速度）
- **匈牙利算法（Hungarian Algorithm）**：在检测框和已有轨迹之间求解最优匹配
- **最近邻匹配（Nearest Neighbor Matching）**：基于外观特征的 Re-ID 匹配，处理遮挡后的重新识别

**双源更新策略**：每帧对跟踪器执行两次 update——先用可见光检测结果更新，再用对齐后的红外检测结果更新。这使得跟踪器能够在可见光条件差（如夜间、强光）时依赖红外检测维持轨迹。

## 性能指标

| 指标 | 目标 |
|---|---|
| 推理延迟 | < 15ms / frame (TensorRT FP16 on Jetson Orin) |
| 端到端延迟 | < 30ms (采集 → 推理 → 跟踪 → 渲染) |
| 帧率 | ≥ 30 FPS (双路 640×480) |
| 检测精度 | mAP@0.5 ≥ 0.55 (COCO val) |

## AI Infra 能力体现

本项目完整覆盖了 AI Infra 工程师的核心能力域：

1. **模型优化与部署** — ONNX 导出 → TensorRT 编译 → FP16 量化 → engine 加载推理
2. **CUDA 高性能编程** — 自定义 kernel 实现预处理/后处理，避免 CPU-GPU 频繁同步
3. **多线程系统设计** — 生产者-消费者流水线，线程安全队列，非阻塞 UI
4. **多传感器集成** — RealSense SDK 集成、相机标定参数管理、跨坐标系变换
5. **边缘端资源优化** — 内存复用、单槽队列、batch 推理、GPU 全链路流水线
6. **工程可维护性** — 模块化架构（camera / algorithm / UI 解耦），CMake 构建系统

## 适用场景

- 安防监控（夜间红外 + 白天可见光全天候检测）
- 工业质检（多光源融合提升缺陷检出率）
- 自动驾驶感知（多传感器融合跟踪）
- 机器人视觉导航（RGB-D + 红外感知）
