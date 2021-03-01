#include "bat.hpp"

Bat::Bat(){
}

Bat::~Bat(){
}

cv::Mat Bat::PrePost(const cv::Mat &src, int color_thresh) {
    cv::Mat binary;
    std::vector<cv::Mat> splited;
    cv::split(src, splited);
    cv::subtract(splited[2], splited[0], binary);
    cv::threshold(binary, binary, color_thresh, 255, cv::THRESH_BINARY);
    cv::Mat struct_elem = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    dilate(binary, binary, struct_elem);
    // binary = temp_binary & gray_binary;
    return binary;
}

void Bat::Predict(const cv::Mat &binary, BatInfo &bat_info) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // get max area rrect
    bool find_flag = false;
    double max_area = 0;
    cv::RotatedRect max_rrect;
    auto contours_size = contours.size();
    for (auto i = 0; i < contours_size; ++i) {
        cv::RotatedRect rrect = cv::minAreaRect(contours[i]);

        double width = MAX(rrect.size.width, rrect.size.height);
        double height = MIN(rrect.size.width, rrect.size.height);
        double ratio = width / height;
        bool condition_one = ratio > config_.min_ratio && ratio < config_.max_ratio;

        double area = rrect.size.area();
        bool condition_two = area > config_.min_area;

        if (condition_two && condition_one) {
            if (area > max_area) {
                max_area = area;
                max_rrect = rrect;
                find_flag = true;
            }
        }
    }

    if (find_flag == false) {
        bat_info.find = false;
    }else {
        bat_info.center_x = max_rrect.center.x;
        bat_info.center_y = max_rrect.center.y;
        max_rrect.points(bat_info.points);
        bat_info.find = true;
    }
    return;
}