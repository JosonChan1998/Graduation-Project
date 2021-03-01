#pragma once
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>

/**
 * @struct FaceInfo  
 * @brief  save the face infomation, x1:left, y1:top, x2:right, y2:bottom
 */
struct FaceInfo {
    double x1;
    double y1;
    double x2;
    double y2;
    double score;
    double landmarks[10];
};

/**
 * @class CenterFace
 * @brief Detect faces of a image
 * @example 
        std::string model_path = "../centerface.onnx";
        CenterFace centerface(model_path,480,640);

        cv::Mat image = cv::imread("../test.jpg", 1);
        std::vector<FaceInfo> face_info;
        centerface.DetectFaces(image,face_info);        
 */
class CenterFace {
public:
    /**
     *@brief  init Centerface model
     *@param  model_path    model path
     *@param  model_input_w model input width
     *@param  model_input_h model input height
     */
    CenterFace(std::string model_path,int model_input_w,int model_input_h,
               bool use_cuda = false);

	~CenterFace();

    /**
     *@brief  Detect faces from a image
     *@param  image         input image
     *@param  faces         faces infomation vector
     *@param  score_thresh  model prob thresh
     */
    void DetectFaces(cv::Mat &image, std::vector<FaceInfo> &faces, 
                     double score_thresh = 0.5, double nms_thresh = 0.3);
private:
    /**
     *@brief The input of model must be multiples of 32, to check the input size
     */
    void CorrectSize(double model_input_w, double model_input_h);

    /**
     *@brief postprocess of the forward output
     */
    void PostProcess(cv::Mat &heatmap, cv::Mat &scale, cv::Mat &offset, 
                     cv::Mat &landmarks, std::vector<FaceInfo>&faces, 
                     double score_thresh, double nms_thresh);

    /**
     *@brief Get the row,col index of heatmap which heatmap's score more than score_thresh
     */
    std::vector<int> get_indexs(float *heatmap, int heatmap_h, 
                                int heatmap_w, double score_thresh);
    /**
     *@brief make all the bbox become square
     */
    void SquareBox(std::vector<FaceInfo>& faces);
    void Nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output, 
             double nms_threshold);
    void Sort(std::vector<FaceInfo>& faces);
private:
    cv::dnn::Net net_;  
    int model_input_h_;    // the correct model height
	int model_input_w_;
	double model_scale_h_; // the correct model scale
	double model_scale_w_;
    int image_h_;     
    int image_w_;
    double img_scale_w_;  // the scale of image_size and model_size
    double img_scale_h_;
};
