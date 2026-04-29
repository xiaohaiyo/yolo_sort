#include "yolov8cu.h"
#include "yolo_datatype.h"
#include "logg.h"

// __device__ 关键字表示只能在 GPU 端调用
__device__ float compute_iou(float cx1, float cy1, float w1, float h1, 
                             float cx2, float cy2, float w2, float h2) {
    float l1 = cx1 - w1 / 2.0f;
    float r1 = cx1 + w1 / 2.0f;
    float t1 = cy1 - h1 / 2.0f;
    float b1 = cy1 + h1 / 2.0f;

    float l2 = cx2 - w2 / 2.0f;
    float r2 = cx2 + w2 / 2.0f;
    float t2 = cy2 - h2 / 2.0f;
    float b2 = cy2 + h2 / 2.0f;

    float inter_l = fmaxf(l1, l2);
    float inter_r = fminf(r1, r2);
    float inter_t = fmaxf(t1, t2);
    float inter_b = fminf(b1, b2);

    if (inter_r <= inter_l || inter_b <= inter_t) return 0.0f;

    float inter_area = (inter_r - inter_l) * (inter_b - inter_t);
    float area1 = w1 * h1;
    float area2 = w2 * h2;

    return inter_area / (area1 + area2 - inter_area);
}

__global__ void preprocess_kernel(uint8_t *pre_d_input, float *pre_d_output, int target_width, int target_height,int or_width,int or_height,float x,float y){
    int idx=blockIdx.x * blockDim.x + threadIdx.x;
    int idy=blockIdx.y * blockDim.y + threadIdx.y;
    int glo_id=idx+idy*blockDim.x*gridDim.x;
    if(idx<target_width&&idy<target_height){
        //新图在原图的位置
        float or_idx = (idx + 0.5f) * x - 0.5f;
        float or_idy = (idy + 0.5f) * y - 0.5f;

        // 2. 边界裁剪 (防止坐标超出原图 0 坐标)
        or_idx = fmaxf(0.0f, or_idx);
        or_idy = fmaxf(0.0f, or_idy);

        // 3. 确定左上角整数坐标 (向下取整)
        int x1 = (int)or_idx;
        int y1 = (int)or_idy;

        // 4. 确定右下角整数坐标 (边界钳位)
        // 建议直接用三目运算符，避开 min 的歧义问题
        int x2 = (x1 + 1 < or_width)  ? x1 + 1 : or_width - 1;
        int y2 = (y1 + 1 < or_height) ? y1 + 1 : or_height - 1;
        //计算权重
        float u=or_idx-x1;
        float v=or_idy-y1;

        float w11=(1-u)*(1-v);
        float w12=u*(1-v);
        float w21=(1-u)*v;
        float w22=u*v;
        //通道转换
        int all=target_width*target_height;

        for(int i=0;i<3;i++){
            //求最终值
            float value_img=
            w11*pre_d_input[(y1*or_width+x1)*3+i]+
            w12*pre_d_input[(y1*or_width+x2)*3+i]+
            w21*pre_d_input[(y2*or_width+x1)*3+i]+
            w22*pre_d_input[(y2*or_width+x2)*3+i];
            float value_infrared=
            w11*pre_d_input[(y1*or_width+x1)*3+i+or_width*or_height*3]+
            w12*pre_d_input[(y1*or_width+x2)*3+i+or_width*or_height*3]+
            w21*pre_d_input[(y2*or_width+x1)*3+i+or_width*or_height*3]+
            w22*pre_d_input[(y2*or_width+x2)*3+i+or_width*or_height*3];
            //归一化加调整为C*H*W
            pre_d_output[i*all+idy*target_width+idx]=value_img/255.0f;
            pre_d_output[i*all+all*3+idy*target_width+idx]=value_infrared/255.0f;
        }
    }
  
}


__global__ void postprocess_kernel1(float *infer_d_output, float *post_d_output, float probabilityThreshold, float nmsThreshold, int topK) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y; // 0代表彩色图，1代表红外图
    int num_boxes = 8400;
    int num_classes = 80;
    if (tid >= num_boxes) return;
// 单张图在输入 TensorRT [Batch, 84, 8400] 中的起始偏移
    int img_offset = batch_id * num_boxes * (4 + num_classes);

    // 单张图在输出 NMS 缓冲区的起始偏移 (1个Count + topK个框 * 7元素)
    int srcArea = 1 + topK * 7; 
    int out_offset = batch_id * srcArea;

    // 1. 寻找 80 个类别中的最大分数
    float max_score = -1.0f;
    int max_class_idx = -1;

    for(int c = 0; c < num_classes; c++) {
        // 读取输入：必须用 tid！
        float raw_score = infer_d_output[img_offset + (4 + c) * num_boxes + tid];
        // float current_score = 1.0f / (1.0f + expf(-raw_score));
        float current_score = raw_score;
        
        
        if(current_score > max_score) {
            max_score = current_score;
            max_class_idx = c;
        }
    }
    // 2. 核心拦截：分数不够直接下班，绝不浪费显存空间
    if (max_score < probabilityThreshold) return;

    int idx = (int)atomicAdd(&post_d_output[out_offset], 1.0f);
    if (idx < topK) {
        // 计算当前这 7 个元素在数组中的确切起始指针
        // out_offset(跳到当前图) + 1(跳过Count头节点) + idx * 7(跳过前面的框)
        float* pout = post_d_output + out_offset + 1 + idx * 7;
        
        // 读取输入用 tid，连续写入 pout
        pout[0] = infer_d_output[img_offset + 0 * num_boxes + tid]; // x
        pout[1] = infer_d_output[img_offset + 1 * num_boxes + tid]; // y
        pout[2] = infer_d_output[img_offset + 2 * num_boxes + tid]; // w
        pout[3] = infer_d_output[img_offset + 3 * num_boxes + tid]; // h
        pout[4] = max_score;                                        // score
        pout[5] = (float)max_class_idx;                             // class_id
        
        // 🚀 极其关键的第 7 元素：初始化生机标志为 1
        pout[6] = 1.0f; 
    } else {
        atomicAdd(&post_d_output[out_offset], -1.0f);
    }
}


__global__ void postprocess_kernel2(float *post_d_output, int topK, float nmsThreshold) {
    // 1. 基础线程信息
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_id = blockIdx.y; 

    // 2. 修正：单图内存体积为 (1个Count + topK*7)
    int srcArea = 1 + topK * 7;
    int img_offset = batch_id * srcArea;

    // 3. 读取有效框数量
    int counts = (int)post_d_output[img_offset];
    int valid_count = counts > topK ? topK : counts;

    // 4. 修正：防止多余线程乱跑，直接下班
    if (tid >= valid_count) return;

    // 🌟 绝招：用指针锁定“我的框”，代码瞬间清爽！
    // 跳过 img_offset 和 开头的 1 个 Count，找到我负责的第 tid 个框
    float* my_box = post_d_output + img_offset + 1 + tid * 7;
    
    // 如果我在上一轮（或者别的操作中）已经被标记死亡，不用比了
    if (my_box[6] == 0.0f) return;

    // 遍历其他框
    for (int i = 0; i < valid_count; i++) {
        if (i == tid) continue; // 不跟自己比

        // 🌟 锁定“对手的框”
        float* other_box = post_d_output + img_offset + 1 + i * 7;

        // 对比 1：类别 (索引 5) 不同，跳过
        if (my_box[5] != other_box[5]) continue;

        // 对比 2：看对方是不是比我强 (分数高，或者同分但编号靠前)
        // [4]是分数
        if (other_box[4] > my_box[4] || (other_box[4] == my_box[4] && i < tid)) {
            
            // 对手比我强，算 IoU！[0][1][2][3] 分别是 x,y,w,h
            float iou = compute_iou(
                my_box[0], my_box[1], my_box[2], my_box[3],
                other_box[0], other_box[1], other_box[2], other_box[3]
            );

            // 对比 3：如果重合度太高，我自杀
            if (iou > nmsThreshold) {
                // 修正：绝不能动 score(4)，只修改专属的生机标志 keep_flag(6)
                my_box[6] = 0.0f;
                return; // 直接 return，结束这个线程，最高效！(比 break 更彻底)
            }
        }
    }
}

void preprocess_cuda(cv::Mat &color_img, cv::Mat &infrared_img, uint8_t *pre_d_input, float *pre_d_output, int target_width, int target_height,int or_width,int or_height){
    dim3  block(32,8);
    dim3  grid((target_width + block.x - 1) / block.x, (target_height + block.y - 1) / block.y);
    float x=(float)or_width/target_width;
    float y=(float)or_height/target_height;
    errCudaCheck(cudaMemcpy(pre_d_input, color_img.data, or_width * or_height * 3, cudaMemcpyHostToDevice));    
    errCudaCheck(cudaMemcpy(pre_d_input+or_width*or_height*3, infrared_img.data, or_width * or_height*3, cudaMemcpyHostToDevice));
    preprocess_kernel<<<grid, block>>>(pre_d_input, pre_d_output, target_width, target_height,or_width,or_height,x,y);
    cudaDeviceSynchronize();

}

void postprocess_cuda(float *infer_d_output,float *post_d_output, float *post_h_output,yolov8Config _config, output_alo &output){
    dim3  block(256);
    dim3  grid((8400 + block.x - 1) / block.x,_config.batchSize);
    // 🚀 必须执行：清理 NMS 缓冲区，重置原子计数器
    int srcArea = 1 + _config.topK * 7;
    int mem_bytes = _config.batchSize * srcArea * sizeof(float);
    errCudaCheck(cudaMemset(post_d_output, 0, mem_bytes));
    postprocess_kernel1<<<grid, block>>>(infer_d_output, post_d_output, _config.probabilityThreshold, _config.nmsThreshold, _config.topK);

    
    dim3 block2(256);
    dim3 grid2((_config.topK + block2.x - 1) / block2.x,_config.batchSize);

    // 调用 kernel2，注意此时传入的是 infer_d_output1 (第一阶段的结果) 作为输入
    postprocess_kernel2<<<grid2, block2>>>(post_d_output, _config.topK, _config.nmsThreshold);
    cudaDeviceSynchronize();
    errCudaCheck(cudaMemcpy(
    post_h_output, // CPU vector 的底层裸指针
    post_d_output,        // GPU 里的 NMS 缓冲区首地址
    mem_bytes,            // 拷贝总字节数
    cudaMemcpyDeviceToHost
));

// ==========================================
    // 6. 🎯 捡子弹：直接在这里解析数据并装填进 _output
    // ==========================================
    // 假设 _config 里面已经存好了原图和目标图的宽高
    float scale_x = (float)_config.or_width / _config.target_width; 
    float scale_y = (float)_config.or_height / _config.target_height;
    // 清空上一帧的历史数据
    output.yolo_output.clear(); 
    

    for (int b = 0; b < _config.batchSize; b++) {
        
        float* current_img_ptr = post_h_output + b * srcArea;
        int counts = std::min((int)current_img_ptr[0], _config.topK);
        
        int valid_boxes_in_this_batch = 0; 

        for (int i = 0; i < counts; i++) {
            float* box_ptr = current_img_ptr + 1 + i * 7;
            
            // 核心判定：只捡起 keep_flag == 1.0f 的存活框
            if (box_ptr[6] == 1.0f) {
                yolo_dection det;
                det.x    = box_ptr[0] * scale_x;
                det.y    = box_ptr[1] * scale_y;
                det.w    = box_ptr[2] * scale_x;
                det.h    = box_ptr[3] * scale_y;
                det.conf = box_ptr[4];
                det.cls  = box_ptr[5]; 
                det.id   = 0;
                
                // 统统塞进扁平化的一维数组里
                output.yolo_output.push_back(det);
                valid_boxes_in_this_batch++;
            }
        }
        
        // 把当前这张图的实际存活数量记录到 dections_sums 里
        output.dections_sums[b] = valid_boxes_in_this_batch;
    }


}

