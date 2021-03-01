#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "../config.hpp"

/**
 * @struct BatInfo  
 * @brief  save the bat info from process result
 * @param  points  minArearrect: left top right bottom
 */
struct BatInfo{
    double center_x;
    double center_y;
    cv::Point2f points[4];
    bool find;
};

/**
 * @class Bat
 * @brief Detect bat in image
 * @example 
        BatInfo bat_info;
        cv::Mat binary = bat_predicter.PrePost(frame, config.color_thresh);
        bat_predicter.Predict(binary, bat_info);      
 */
class Bat {
private:
    Config config_;
public:
    Bat();
    ~Bat(); 
    cv::Mat PrePost(const cv::Mat &src, int color_thresh);
    void Predict(const cv::Mat &binary, BatInfo &bat_info);
};
