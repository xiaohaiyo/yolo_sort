
#include "yolo_datatype.h"
#include "opencv2/opencv.hpp"
void postprocess_cuda(float *infer_d_output,float *post_d_output, float *post_h_output,yolov8Config _config,output_alo &_output);
void preprocess_cuda(cv::Mat &color_img, cv::Mat &infrared_img, uint8_t *pre_d_input, float *pre_d_output, int target_width, int target_height,int or_width,int or_height);
