#pragma once
#include <iostream>
#include <string>

enum Mode {
    FACE = 0,
    BAT = 1,
};

class Config {
public:
    Config() {
        mode = FACE;
        serail_path = "/dev/ttyUSB0";
        camera_path = "/dev/video0";
        buffer_size = 3;
        camera_w = 640;
        camera_h = 480;
        exposure = 500;
        centerface_path = "../checkpoints/centerface.onnx";
        model_w = 640;
        model_h = 480;
        score_thresh = 0.5;
        nms_thresh = 0.3;
        color_thresh = 75;
        min_ratio = 0.5;
        max_ratio = 1.5;
        min_area = 1500;
    }
public:
    // init
    int mode;
    std::string serail_path;
    std::string camera_path;
    int buffer_size;
    int camera_w;
    int camera_h;
    int exposure;
    std::string centerface_path;
    int model_w;
    int model_h;
    // param
    double score_thresh;
    double nms_thresh;
    int color_thresh;
    double min_ratio;
    double max_ratio;
    int min_area;
};