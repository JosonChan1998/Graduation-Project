#include "img_prod_cons.hpp"

ImageConsProd::ImageConsProd() {
    prd_idx_ = 0;
    csm_idx_ = 0;
    data_->frame = 0 ;
}

void ImageConsProd::ImageProducer() {
    RMVideoCapture cap("/dev/video0", 1);
    if (cap.fd != -1) {
        cap.setVideoFormat(640, 480, 1);
        cap.setExposureTime(0, 157);
        cap.startStream();
        cap.info();
        std::cout << "camera open sucess !!" << std::endl;
    }else {
        std::cout << "camera init failed !!" << std::endl;
        return;
    }

    while (true) {
        while (prd_idx_ - csm_idx_ >= BUFFER_SIZE);
        cv::Mat img;
        auto start = std::chrono::steady_clock::now();
        cap >> img;
        auto end = std::chrono::steady_clock::now();
        std::cout << "get image:" 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << " ms " << std::endl;
        data_[prd_idx_ % BUFFER_SIZE].img = img;
        data_[prd_idx_ % BUFFER_SIZE].frame++;
        prd_mtx_.lock();
        ++prd_idx_;
        prd_mtx_.unlock();
    }
}

void ImageConsProd::ImageConsumer() {
    SerialPort port("/dev/ttyUSB0");
    bool status = port.initSerialPort();
    if (status == false) {
        std::cout << "serail port init failed !!" << std::endl;
        return;
    }else {
        std::cout << "open serail port sucess !!" << std::endl;
    }

    CenterFace centerface("../checkpoints/centerface.onnx", 640, 480, false);
    Bat bat_predicter;
    std::cout << "model init sucess !!" << std::endl;

    while (true) {
        while(prd_idx_ - csm_idx_ == 0);
        cv::Mat frame;
        data_[csm_idx_ % BUFFER_SIZE].img.copyTo(frame);
        csm_mtx_.lock();
        ++csm_idx_;
        csm_mtx_.unlock();

        auto tp1 = std::chrono::steady_clock::now();
        std::vector<FaceInfo> face_info;
        centerface.DetectFaces(frame, face_info);
        auto tp2 = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count() << " ms " << std::endl;

        VisionData data = {0, 0, 0, 0};
        if (!face_info.empty()) {
            double center_x = (face_info[0].x1 + face_info[0].x2) / 2;
            double center_y = (face_info[0].y1 + face_info[0].y2) / 2;
            data = {center_x, center_y, 0, 1};
        }
        port.TransformData(data);
        port.send();
    }
}