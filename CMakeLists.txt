cmake_minimum_required(VERSION 3.16)
project(my_app LANGUAGES CXX CUDA)

# --- 1. 全局设置 & 找包 (必须放在 add_library 之前) ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# 查找 CUDA
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
    set(CUDA_LIBRARIES "/usr/local/cuda/lib64/libcudart.so")
endif()

# 查找 Eigen (提前到这里！)
set(Eigen3_DIR "/home/nvidia/fu_projext/orin_gtx_yolov8/eigen-3.3.9")
find_package(Eigen3 REQUIRED)

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets Core)

# --- 2. 处理 CUDA 的编译参数 (针对 Orin) ---
string(REPLACE "-fPIC" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_COMPILE_OPTIONS_PIC "")

# --- 3. 定义算法库 alo (因为它被 d45i 包含) ---
file(GLOB_RECURSE SRC_CPP
    "${CMAKE_SOURCE_DIR}/algorithm/alg_pipeline/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/algorithm/yolo/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/algorithm/sort/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/algorithm/sort/3rdparty/deepsort/src/*.cpp"
)
file(GLOB_RECURSE SRC_CU "${CMAKE_SOURCE_DIR}/algorithm/yolo/src/*.cu")

add_library(alo SHARED ${SRC_CPP} ${SRC_CU})
set_target_properties(alo PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
)#意思是自动处理 Qt 5 的资源文件
set_target_properties(alo PROPERTIES POSITION_INDEPENDENT_CODE OFF CUDA_ARCHITECTURES 87)
# 1. 针对 alo 这个目标，先关闭自动的 PIC
set_target_properties(alo PROPERTIES POSITION_INDEPENDENT_CODE OFF)

# 2. 手动根据编译语言分发参数
target_compile_options(alo PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>           # 如果是 C++，正常加 -fPIC
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC> # 如果是 CUDA，通过 -Xcompiler 转发给底层的 gcc
)
# 关键：给 alo 设置包含路径，并设为 PUBLIC，这样链接它的库能自动获取路径
target_include_directories(alo PUBLIC
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/algorithm/alg_pipeline/include
    ${CMAKE_SOURCE_DIR}/algorithm/yolo/include
    ${CMAKE_SOURCE_DIR}/algorithm/sort/include
    ${CMAKE_SOURCE_DIR}/algorithm/sort/3rdparty/deepsort/include
    ${CMAKE_SOURCE_DIR}/algorithm/yolo/common
     ${CMAKE_SOURCE_DIR}/algorithm/common
      ${CMAKE_SOURCE_DIR}/common/include

    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(alo PUBLIC 
    Eigen3::Eigen 
    ${OpenCV_LIBRARIES} 
    nvinfer nvonnxparser 
    ${CUDA_LIBRARIES}
    Qt5::Core      # ⭐ 必须加上这个，否则编译器拿不到 QString 的路径
    Qt5::Widgets   # 如果你的数据结构里涉及界面相关类，也加上这个
)

# --- 4. 定义 D45i 库 ---
file(GLOB_RECURSE SRC_D453I "${CMAKE_SOURCE_DIR}/d453i/src/*.cpp")

add_library(d45i SHARED ${SRC_D453I})

# 关键修复：d45i 必须链接 alo，因为它的代码包含了 alo 的头文件
target_link_libraries(d45i PUBLIC 
    alo 
    realsense2::realsense2 
    Qt5::Core
)

target_include_directories(d45i PUBLIC
    ${CMAKE_SOURCE_DIR}/d453i/include
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_BINARY_DIR}                  # ⭐ 必须加这一行，让编译器找到 moc_mainwindow.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/my_app_autogen/include # 如果
)

# --- 5. 主程序 ---
add_executable(my_app 
    ${CMAKE_SOURCE_DIR}/src/main.cpp 
    ${CMAKE_SOURCE_DIR}/src/mainwindow.cpp 
    ${CMAKE_SOURCE_DIR}/include/mainwindow.h
)

# 🚨 重点在这里：确保包含路径覆盖了头文件所在的目录
target_include_directories(my_app PRIVATE 
    ${CMAKE_SOURCE_DIR}/include          # 👈 如果 mainwindow.h 在这个文件夹里，必须加这行
    ${CMAKE_SOURCE_DIR}/d453i/include
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}          # 确保 Qt 自动生成的 moc 文件能被找到
)

target_link_libraries(my_app PRIVATE Qt5::Widgets d45i)