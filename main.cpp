#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "bat.hpp"
#include "config.hpp"
#include "face_api.hpp"
#include "serialport.h"
#include "RMVideoCapture.h"

int main(int argc, char** argv) {
    Config config;

    RMVideoCapture cap(config.camera_path, config.buffer_size);
    if (cap.fd != -1) {
        cap.setVideoFormat(config.camera_w, config.camera_h, 1);
        cap.setExposureTime(0, config.exposure);
        cap.startStream();
        std::cout << "camera open sucess !!" << std::endl;
    }else {
        std::cout << "camera init failed !!" << std::endl;
        return -1;
    }

    SerialPort port(config.serail_path);
    bool status = port.initSerialPort();
    if (status == false) {
        std::cout << "serail port init failed !!" << std::endl;
        return -1;
    }else {
        std::cout << "open serail port sucess !!" << std::endl;
    }

    CenterFace centerface(config.centerface_path, 
                          config.model_w, config.model_h, false);
    Bat bat_predicter;
    std::cout << "model init sucess !!" << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.size().empty()) {
            continue;
        }
        if (config.mode == FACE) {
            auto tp1 = std::chrono::steady_clock::now();
            std::vector<FaceInfo> face_info;
            centerface.DetectFaces(frame, face_info, config.score_thresh, config.nms_thresh);
            auto tp2 = std::chrono::steady_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count() << " ms " << std::endl;

            VisionData data = {0, 0, 0, 0};
            if (!face_info.empty()) {
                double center_x = (face_info[0].x1 + face_info[0].x2) / 2;
                double center_y = (face_info[0].y1 + face_info[0].y2) / 2;
                data = {static_cast<float> (center_x), 
                        static_cast<float> (center_y), 0, 1};
            }
            port.TransformData(data);
            port.send();

            if (!face_info.empty()) {
                for (int i = 0; i < 1; i++) {
                    cv::rectangle(frame, cv::Point(face_info[i].x1, face_info[i].y1), 
                                cv::Point(face_info[i].x2, face_info[i].y2), 
                                cv::Scalar(0, 255, 0), 2);
                    for (int j = 0; j < 5; j++) {
                        cv::circle(frame, cv::Point(face_info[i].landmarks[2*j], 
                                face_info[i].landmarks[2*j+1]), 2, 
                                cv::Scalar(255, 255, 0), 1);
                    }
                }
            }
            cv::imshow("image",frame);
            cv::waitKey(1);
        }else if(config.mode == BAT) {
            BatInfo bat_info;
            cv::Mat binary = bat_predicter.PrePost(frame, config.color_thresh);
            bat_predicter.Predict(binary, bat_info);

            VisionData data = {0, 0, 0, 0};
            if (bat_info.find == true)
                data = {static_cast<float> (bat_info.center_x), 
                        static_cast<float> (bat_info.center_y), 0, 1};
            port.TransformData(data);
            port.send();

            for (int j = 0; j < 4; j++){
                cv::line(frame, bat_info.points[j], bat_info.points[(j + 1) % 4], 
                         cv::Scalar(0, 255, 0), 3);
            }
            cv::imshow("image",frame);
            cv::imshow("binary",binary);
            cv::waitKey(1);
        }
    }
    return 0;
}