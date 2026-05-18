# 项目环境依赖说明

## 1. 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|---|---|---|
| GPU | NVIDIA Jetson Orin (SM 8.7) | Jetson Orin NX 16GB / AGX Orin 32GB |
| 深度相机 | Intel RealSense D453i | Intel RealSense D453i |
| 内存 | 8GB | 16GB+ |
| 存储 | 16GB 可用空间 | 32GB+ (含模型文件) |

## 2. 操作系统

| 项目 | 版本 |
|---|---|
| 系统 | Ubuntu 20.04 / 22.04 (aarch64) |
| JetPack | 5.1+ / 6.0+ |
| 内核 | Linux 5.10+ |

## 3. 软件依赖

### 3.1 核心依赖

| 依赖 | 最低版本 | 推荐版本 | 用途 |
|---|---|---|---|
| CUDA Toolkit | 11.4 | 12.2+ | GPU 并行计算、自定义 kernel |
| cuDNN | 8.4 | 8.9+ | 深度学习加速库 |
| TensorRT | 8.4 | 8.6+ | 模型推理优化引擎 |
| OpenCV | 4.5 | 4.8+ | 图像处理、色彩转换、去畸变 |
| Qt5 | 5.12 | 5.15 | GUI 界面框架 |
| librealsense2 | 2.50 | 2.54+ | RealSense 相机驱动与数据采集 |
| Eigen | 3.3.9 | 3.4+ | 线性代数（卡尔曼滤波、矩阵运算） |
| CMake | 3.16 | 3.22+ | 构建系统 |

### 3.2 编译工具链

| 工具 | 版本 |
|---|---|
| GCC / G++ | 9.4+ (支持 C++17) |
| NVCC | CUDA Toolkit 自带 |

## 4. 安装指南 (Jetson Orin)

### 4.1 CUDA / cuDNN / TensorRT

JetPack SDK 会预装 CUDA、cuDNN 和 TensorRT。确认安装：

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 TensorRT 版本
dpkg -l | grep tensorrt

# 检查 cuDNN 版本
dpkg -l | grep cudnn
```

如需手动安装 TensorRT：
```bash
# 通过 NVIDIA apt 源安装
sudo apt-get update
sudo apt-get install -y tensorrt
sudo apt-get install -y libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev
```

### 4.2 OpenCV

Jetson 通常预装 OpenCV，确认版本：
```bash
pkg-config --modversion opencv4
```

如需从源码编译（推荐使用 Jetson 专用脚本）：
```bash
# 安装依赖
sudo apt-get install -y build-essential cmake git libgtk2.0-dev \
    pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev

# 下载并编译 OpenCV 4.8
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
unzip opencv.zip && cd opencv-4.8.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN="8.7" \
      -D WITH_GSTREAMER=ON ..
make -j$(nproc)
sudo make install
```

### 4.3 Qt5

```bash
sudo apt-get install -y qt5-default qtbase5-dev qtchooser \
    qt5-qmake qtbase5-dev-tools
```

> Ubuntu 22.04 中 `qt5-default` 已移除，使用：
> ```bash
> sudo apt-get install -y qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
> ```

### 4.4 librealsense2

```bash
# 方法一：通过 apt 安装
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt-get update
sudo apt-get install -y librealsense2-dev librealsense2-utils

# 方法二：从源码编译（推荐 Jetson）
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_EXAMPLES=ON \
         -DBUILD_WITH_CUDA=ON
make -j$(nproc)
sudo make install
```

验证安装：
```bash
realsense-viewer
```

### 4.5 Eigen

```bash
# 方法一：通过 apt 安装
sudo apt-get install -y libeigen3-dev

# 方法二：下载指定版本（项目使用 3.3.9）
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
tar -xzf eigen-3.3.9.tar.gz
# 修改 CMakeLists.txt 中的 Eigen3_DIR 指向解压路径
```

### 4.6 模型文件准备

```bash
# 1. 下载 YOLOv8n 预训练模型
pip install ultralytics
yolo export model=yolov8n.pt format=onnx

# 2. 将 ONNX 转换为 TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolov8n.onnx \
    --saveEngine=yolov8n.engine \
    --fp16 \
    --workspace=1024

# 3. 将文件放到指定路径（或修改 yolo_datatype.h 中的路径）
mkdir -p input/
mv yolov8n.onnx input/
mv yolov8n.engine input/
```

## 5. 环境变量

```bash
# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# TensorRT（如手动安装）
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Eigen（如手动安装）
export Eigen3_DIR=/path/to/eigen-3.3.9
```

建议写入 `~/.bashrc` 使其持久化。

## 6. 依赖版本兼容矩阵

| JetPack | CUDA | cuDNN | TensorRT | OpenCV | GCC |
|---|---|---|---|---|---|
| 5.1.1 | 11.4 | 8.4 | 8.4 | 4.5 | 9.4 |
| 5.1.2 | 11.4 | 8.5 | 8.5 | 4.5 | 9.4 |
| 6.0 | 12.2 | 8.9 | 8.6 | 4.8 | 11.4 |

> 建议使用 JetPack 5.1.2 或 6.0，已验证兼容性最佳。

## 7. 常见问题

### Q1: TensorRT 找不到 `libnvinfer.so`
```bash
# 检查库路径
find / -name "libnvinfer.so" 2>/dev/null
# 添加到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/tensorrt/lib:$LD_LIBRARY_PATH
```

### Q2: CUDA 架构不匹配
```bash
# 确认 GPU 架构
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Jetson Orin 为 8.7，确保 CMakeLists.txt 中 CUDA_ARCHITECTURES 设置正确
```

### Q3: Qt5 MOC 文件找不到
```bash
# 确保 CMake 开启了自动 MOC
set(CMAKE_AUTOMOC ON)
# 确保包含 build 目录
target_include_directories(my_app PRIVATE ${CMAKE_BINARY_DIR})
```

### Q4: RealSense 相机无法识别
```bash
# 检查 USB 连接
lsusb | grep Intel
# 检查固件版本
rs-enumerate-devices | grep -i firmware
# 更新固件（如需要）
realsense-viewer  # 在 GUI 中更新
```

### Q5: Eigen 头文件找不到
```bash
# 方法一：通过 apt 安装到系统路径
sudo apt-get install libeigen3-dev

# 方法二：修改 CMakeLists.txt 中 Eigen3_DIR 为实际路径
set(Eigen3_DIR "/your/path/to/eigen-3.3.9")
```
