/*******************************************************************************************************************
Copyright 2017 Dajiang Innovations Technology Co., Ltd (DJI)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files(the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
*******************************************************************************************************************/

#include "RMVideoCapture.h"
#include "linux/videodev2.h"

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

RMVideoCapture::RMVideoCapture(const char * device, int size_buffer) : video_path(device) {
    fd = open(device, O_RDWR);
    buffer_size = size_buffer;
    buffr_idx = 0;
    cur_frame = 0;
    capture_width = 0;
    capture_height = 0;
    mb = new MapBuffer[buffer_size];
}

RMVideoCapture::RMVideoCapture(std::string device, int size_buffer) {
    const char *device_c = device.c_str();
    video_path = device_c;
    fd = open(device_c, O_RDWR);
    buffer_size = size_buffer;
    buffr_idx = 0;
    cur_frame = 0;
    capture_width = 0;
    capture_height = 0;
    mb = new MapBuffer[buffer_size];
}

RMVideoCapture::~RMVideoCapture(){
    close(fd);
    delete [] mb;
}

/**
  * @brief restart the device
  * @param  none
  * @return none
  */
void RMVideoCapture::restartCapture(){
    close(fd);
    fd = open(video_path, O_RDWR);
    buffr_idx = 0;
    cur_frame = 0;
}

/**
  * @brief change the image to Mat
  * @param  void *data
  * @param  image:the image
  * @return bool
  */
void RMVideoCapture::cvtRaw2Mat(const void * data, cv::Mat & image){
    if (format == V4L2_PIX_FMT_MJPEG){
        // because of the imdecode source code in opencv, you need the one channel, not 3 channels
        // cv::Mat src(capture_height, capture_width, CV_8UC3, (void*) data);
        cv::Mat src(1, capture_height * capture_width * 3, CV_8U, (void*)data);
        image = cv::imdecode(src, 1);
    }
    else if(format == V4L2_PIX_FMT_YUYV){
        cv::Mat yuyv(capture_height, capture_width, CV_8UC2, (void*) data);
        cv::cvtColor(yuyv, image, cv::COLOR_YUV2BGR_YUYV);
    }
}

/**
  * @brief get the image
  * @param  image:the image of Mat
  * @return none
  */
RMVideoCapture & RMVideoCapture::operator >> (cv::Mat & image) {
//    std::cout << "current buffr idx: " << buffr_idx << std::endl;
    struct v4l2_buffer bufferinfo = {0};
    bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufferinfo.memory = V4L2_MEMORY_MMAP;
    bufferinfo.index = buffr_idx;
    if(ioctl(fd, VIDIOC_DQBUF, &bufferinfo) < 0){
        perror("VIDIOC_DQBUF Error");
        exit(1);
    }
    // std::cout << "raw data size: " << bufferinfo.bytesused << std::endl;
    cvtRaw2Mat(mb[buffr_idx].ptr, image);

    //memV(&bufferinfo, 0, sizeof(bufferinfo));
    bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufferinfo.memory = V4L2_MEMORY_MMAP;
    bufferinfo.index = buffr_idx;

    // Queue the next one.
    if(ioctl(fd, VIDIOC_QBUF, &bufferinfo) < 0){
        perror("VIDIOC_DQBUF Error");
        exit(1);
    }
    ++buffr_idx;
    buffr_idx = buffr_idx >= buffer_size ? buffr_idx - buffer_size : buffr_idx;
    ++cur_frame;
    return *this;
}

/**
  * @brief init MMAP and put the buffer to query
  * @param  none
  * @return bool
  */
bool RMVideoCapture::initMMap(){
    /*
     申请和管理缓冲区，应用程序和设备有三种交换数据的方法，直接read/write ，内存映射(memorymapping) ，用户指针。这里只讨论 memorymapping.
     函数   ：int ioctl(intfd, int request, struct v4l2_requestbuffers *argp);
     struct v4l2_requestbuffers  申请帧缓冲
     {
        __u32 count;            // 缓冲区内缓冲帧的数目
        enum v4l2_buf_type type;// 缓冲帧数据格式
        enum v4l2_memorymemory; // 区别是内存映射还是用户指针方式
        __u32 reserved[2];
     };
     VIDIOC_REQBUFS         //指令含义：分配内存
     */
    /*  fd ：文件描述符 */
    struct v4l2_requestbuffers bufrequest = {0};
    bufrequest.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufrequest.memory = V4L2_MEMORY_MMAP;
    bufrequest.count = buffer_size;

    if(ioctl(fd, VIDIOC_REQBUFS, &bufrequest) < 0){
        perror("VIDIOC_REQBUFS");
        return false;
    }


    /*
     函数   ：int ioctl(intfd, int request, struct v4l2_buffer *argp);
     struct v4l2_buffer
     {
        __u32 index;                //buffer 序号
        enum v4l2_buf_type type;    //buffer 类型
        __u32 byteused;             //buffer 中已使用的字节数
        __u32 flags;                //区分是MMAP 还是USERPTR
        enum v4l2_fieldfield;
        struct timeval timestamp;   //获取第一个字节时的系统时间
        struct v4l2_timecode timecode;
        __u32 sequence;             //队列中的序号
        enum v4l2_memory memory;    //IO 方式，被应用程序设置
        union m
        {
            __u32 offset;           //缓冲帧地址，只对MMAP 有效
            unsigned longuserptr;
        };
        __u32 length;               //缓冲帧长度
        __u32 input;
        __u32 reserved;
     };
     VIDIOC_QUERYBUF        //把VIDIOC_REQBUFS中分配的数据缓存转换成物理地址
     */
    for(int i = 0; i < buffer_size; ++i){
        struct v4l2_buffer bufferinfo = {0};
        bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        bufferinfo.memory = V4L2_MEMORY_MMAP;
        bufferinfo.index = i; /* Queueing buffer index 0. */

        // Put the buffer in the incoming queue.
        if(ioctl(fd, VIDIOC_QUERYBUF, &bufferinfo) < 0){
            perror("VIDIOC_QUERYBUF");
            return false;
        }

        /*
         #include<sys/mman.h>
         void *mmap(void*addr, size_t length, int prot, int flags, int fd, off_t offset);
         //addr 映射起始地址，一般为NULL ，让内核自动选择
         //length 被映射内存块的长度
         //prot 标志映射后能否被读写，其值为PROT_EXEC,PROT_READ,PROT_WRITE,PROT_NONE
         //flags 确定此内存映射能否被其他进程共享，MAP_SHARED,MAP_PRIVATE
         //fd,offset, 确定被映射的内存地址
         成功：返回成功映射后的地址，不成功：返回MAP_FAILED ((void*)-1);

         int munmap(void*addr, size_t length);// 断开映射
         //addr 为映射后的地址，length 为映射后的内存长度
         */
        /*  fd ：文件描述符 addr：二级指针，用来存放开辟的buffers地址 */
        mb[i].ptr = mmap(
            NULL,
            bufferinfo.length,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            bufferinfo.m.offset);
        mb[i].size = bufferinfo.length;

        if(mb[i].ptr == MAP_FAILED){
            perror("MAP_FAILED");
            return false;
        }
        memset(mb[i].ptr, 0, bufferinfo.length);

        // 将申请到的缓冲帧放入队
        memset(&bufferinfo, 0, sizeof(bufferinfo));
        bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        bufferinfo.memory = V4L2_MEMORY_MMAP;
        bufferinfo.index = i;
        if(ioctl(fd, VIDIOC_QBUF, &bufferinfo) < 0){
            perror("VIDIOC_QBUF");
            return false;
        }
    }
    return true;
}

/**
  * @brief to start catch the stream
  * @param  none
  * @return bool
  */
bool RMVideoCapture::startStream(){

    /*
     VIDIOC_STREAMON    指令含义：开始采集
     VIDIOC_STREAMOFF    指令含义：结束采集
     */
    /*  fd ：文件描述符 */
    cur_frame = 0;
    refreshVideoFormat();
    if(initMMap() == false)
        return false;

    __u32 type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(ioctl(fd, VIDIOC_STREAMON, &type) < 0){
        perror("VIDIOC_STREAMON");
        return false;
    }
    return true;
}

/**
  * @brief to close catch the stream
  * @param  none
  * @return bool
  */
bool RMVideoCapture::closeStream(){
    cur_frame = 0;
    buffr_idx = 0;
    __u32 type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(ioctl(fd, VIDIOC_STREAMOFF, &type) < 0){
        perror("VIDIOC_STREAMOFF");
        return false;
    }
    for(int i = 0; i < buffer_size; ++i){
        munmap(mb[i].ptr, mb[i].size);
    }
    return true;
}

/**
  * @brief to set the exposure time
  * @param  auto_exp: 0
  * @param  t: the exposure value
  * @return bool
  */
bool RMVideoCapture::setExposureTime(bool auto_exp, int t){
    if (auto_exp){
        struct v4l2_control control_s;
        control_s.id = V4L2_CID_EXPOSURE_AUTO;
        control_s.value = V4L2_EXPOSURE_AUTO;
        if( xioctl(fd, VIDIOC_S_CTRL, &control_s) < 0){
            printf("Set Auto Exposure error\n");
            return false;
        }
    }
    else {
        struct v4l2_control control_s;
        control_s.id = V4L2_CID_EXPOSURE_AUTO;
        control_s.value = 1;
        if( xioctl(fd, VIDIOC_S_CTRL, &control_s) < 0){
            printf("Close MANUAL Exposure error\n");
//            return false;
        }

        control_s.id = V4L2_CID_EXPOSURE_ABSOLUTE;
        control_s.value = t;
        if( xioctl(fd, VIDIOC_S_CTRL, &control_s) < 0){
            printf("Set Exposure Time error\n");
//            return false;
        }
    }
    return true;
}

/**
  * @brief to change the Video Format and restart the device
  * @param  width
  * @param  height
  * @param  mijg: to choose YUV or MJPG
  * @return bool
  */
bool RMVideoCapture::changeVideoFormat(int width, int height, bool mjpg){
    closeStream();
    restartCapture();
    setVideoFormat(width, height, mjpg);
    startStream();
    return true;
}

/**
  * @brief to set the Video Format
  * @param  width
  * @param  height
  * @param  mijg: to choose YUV or MJPG
  * @return bool
  */
bool RMVideoCapture::setVideoFormat(int width, int height, bool mjpg){
    /*struct v4l2_format fmt = {
                    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
                    .fmt = {
                .pix = {
                                    .width = w,                         //帧宽
                                    .height = h,                        //帧高
                                    .pixelformat = V4L2_PIX_FMT_MJPEG,  //帧格式
                                    .field = V4L2_FIELD_ANY,
                            }
                    },
        };*/
    if (capture_width == width && capture_height == height)
        return true;
    capture_width = width;
    capture_height = height;
    cur_frame = 0;
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    if (mjpg == true)
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    else
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt)){
        printf("Setting Pixel Format\n");
        return false;
    }
    return true;
}

/**
  * @brief  refresh the Video Format
  * @param  none
  * @return bool
  */
bool RMVideoCapture::refreshVideoFormat(){
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt)) {
        perror("Querying Pixel Format\n");
        return false;
    }
    capture_width = fmt.fmt.pix.width;
    capture_height = fmt.fmt.pix.height;
    format = fmt.fmt.pix.pixelformat;
    return true;
}

/**
  * @brief  to get the videoSize
  * @param  width
  * @param  height
  * @return bool
  */
bool RMVideoCapture::getVideoSize(int & width, int & height){
    if (capture_width == 0 || capture_height == 0){
        if (refreshVideoFormat() == false)
            return false;
    }
    width = capture_width;
    height = capture_height;
    return true;
}

/**
  * @brief  to set the fps
  * @param  fps
  * @return bool
  */
bool RMVideoCapture::setVideoFPS(int fps){
    struct v4l2_streamparm stream_param = {0};
    stream_param.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    stream_param.parm.capture.timeperframe.denominator = fps;
    stream_param.parm.capture.timeperframe.numerator = 1;

    if (-1 == xioctl(fd, VIDIOC_S_PARM, &stream_param)){
        printf("Setting Frame Rate Fail\n");
        return false;
    }
    return true;
}

/**
  * @brief  to set the bufferSize
  * @param  bsize:buffer_size
  * @return bool
  */
bool RMVideoCapture::setBufferSize(int bsize){
    if (buffer_size != bsize){
        buffer_size = bsize;
        delete [] mb;
        mb = new MapBuffer[buffer_size];
    }
}

/**
  * @brief  to get the camera info
  * @param  none
  * @return bool
  */
void RMVideoCapture::info(){
    /*
     头文件 ：<linux/videodev2.h>
     函数   ：int ioctl(intfd, int request, struct v4l2_capability *argp);
     struct v4l2_capability     设备的功能
     {
        __u8 driver[16];   // 驱动名字
        __u8 card[32];     // 设备名字
        __u8 bus_info[32]; // 设备在系统中的位置
        __u32 version;     // 驱动版本号
        __u32 capabilities;// 设备支持的操作
        __u32 reserved[4]; // 保留字段
     };
     VIDIOC_QUERYCAP    //查询驱动功能
     */
    /*  fd：文件描述符  */
    struct v4l2_capability caps = {};
    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &caps)) {
            perror("Querying Capabilities\n");
            return;
    }

    printf( "Driver Caps:\n"
            "  Driver: \"%s\"\n"
            "  Card: \"%s\"\n"
            "  Bus: \"%s\"\n"
            "  Version: %d.%d\n"
            "  Capabilities: %08x\n",
            caps.driver,
            caps.card,
            caps.bus_info,
            (caps.version>>16)&&0xff,
            (caps.version>>24)&&0xff,
            caps.capabilities);
    camnum = caps.bus_info[17] -48;
    std::cout<<"------cam-----number----:"<<camnum<<std::endl;

/*    struct v4l2_cropcap

    {

    enum v4l2_buf_type type; // 数据流的类型，应用程序设置

    struct v4l2_rect bounds; // 这是 camera 的镜头能捕捉到的窗口大小的局限

    struct v4l2_rect defrect; // 定义默认窗口大小，包括起点位置及长,宽的大小，大小以像素为单位

    struct v4l2_fract pixelaspect; // 定义了图片的宽高比

    };*/
    struct v4l2_cropcap cropcap = {0};
    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl (fd, VIDIOC_CROPCAP, &cropcap))  {
            perror("Querying Cropping Capabilities\n");
            return;
    }

    printf( "Camera Cropping:\n"
            "  Bounds: %dx%d+%d+%d\n"
            "  Default: %dx%d+%d+%d\n"
            "  Aspect: %d/%d\n",
            cropcap.bounds.width, cropcap.bounds.height, cropcap.bounds.left, cropcap.bounds.top,
            cropcap.defrect.width, cropcap.defrect.height, cropcap.defrect.left, cropcap.defrect.top,
            cropcap.pixelaspect.numerator, cropcap.pixelaspect.denominator);

    /*
     函数   ：int ioctl(intfd, int request, struct v4l2_fmtdesc *argp);
     struct v4l2_fmtdesc
     {
        __u32 index;               // 要查询的格式序号，应用程序设置
        enum v4l2_buf_type type;   // 帧类型，应用程序设置
        __u32 flags;               // 是否为压缩格式
        __u8 description[32];      // 格式名称
        __u32 pixelformat;         // 格式
        __u32 reserved[4];         // 保留
     };
     VIDIOC_ENUM_FMT    //指令含义：获取当前驱动支持的视频格式
     */
    /*  fd：文件描述符  */
    struct v4l2_fmtdesc fmtdesc = {0};
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    char fourcc[5] = {0};
    char c, e;
    printf("  FMT : CE Desc\n--------------------\n");
    while (0 == xioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc)) {
            strncpy(fourcc, (char *)&fmtdesc.pixelformat, 4);
            c = fmtdesc.flags & 1? 'C' : ' ';
            e = fmtdesc.flags & 2? 'E' : ' ';
            printf("  %s: %c%c %s\n", fourcc, c, e, fmtdesc.description);
            fmtdesc.index++;
    }


    /*
     函数   ：int ioctl(intfd, int request, struct v4l2_format *argp);
     struct v4l2_format     帧的格式
     {
        enum v4l2_buf_type type;        //帧类型，应用程序设置
       union fmt
        {
            structv4l2_pix_format pix;  //视频设备使用
            structv4l2_window win;
            structv4l2_vbi_format vbi;
            structv4l2_sliced_vbi_format sliced;
            __u8raw_data[200];
        };
     };
     VIDIOC_G_FMT:      //指令含义：读取当前驱动的捕获格式
     VIDIOC_S_FMT:      //指令含义：设置当前驱动的捕获格式
     VIDIOC_ENUM_FMT:   //指令含义：获取当前驱动支持的视频格式
     */
    /*  fd：文件描述符  */
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt)) {
        perror("Querying Pixel Format\n");
        return;
    }
    strncpy(fourcc, (char *)&fmt.fmt.pix.pixelformat, 4);
    printf( "Selected Camera Mode:\n"
            "  Width: %d\n"
            "  Height: %d\n"
            "  PixFmt: %s\n"
            "  Field: %d\n",
            fmt.fmt.pix.width,
            fmt.fmt.pix.height,
            fourcc,
            fmt.fmt.pix.field);

    // 得到帧率信息
    struct v4l2_streamparm streamparm = {0};
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == ioctl(fd, VIDIOC_G_PARM, &streamparm)) {
        perror("Querying Frame Rate\n");
        return;
    }
    printf( "Frame Rate:  %f\n====================\n",
            (float)streamparm.parm.capture.timeperframe.denominator /
            (float)streamparm.parm.capture.timeperframe.numerator);
}

bool RMVideoCapture::setSaturation(int Saturation){
    struct v4l2_control SATURATION;
    SATURATION.id = V4L2_CID_SATURATION;
    SATURATION.value = Saturation;
    if(xioctl(fd,VIDIOC_S_CTRL,&SATURATION) < 0)
    {
        perror("can't set the satutation\n");
        return false;
    }
    return true;
}

/**
  * @brief  chongzai ioctl
  * @param  fd: the video_file
  * @param  request:iotcl type
  * @param  arg:the value
  * @return bool
  */
int RMVideoCapture::xioctl(int fd, int request, void *arg){
    int r;
    do r = ioctl (fd, request, arg);
    while (-1 == r && EINTR == errno);
    return r;
}
