#pragma once
#include <iostream>
#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "RMVideoCapture.h"
#include "face_api.hpp"
#include "serialport.h"
#include "bat.hpp"

#define BUFFER_SIZE 1

struct ImageData {
    cv::Mat img;
    unsigned int frame;
};

class ImageConsProd {
public:
    ImageConsProd();
    void ImageProducer();
    void ImageConsumer();
private:
    std::mutex prd_mtx_;
    std::mutex csm_mtx_;
    volatile unsigned int prd_idx_;
    volatile unsigned int csm_idx_;
    ImageData data_[BUFFER_SIZE];
};