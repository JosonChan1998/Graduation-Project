# project name
PROJECT(graduation_project)
cmake_minimum_required(VERSION 3.5)

# set the compiler option
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# directory of src headers
include_directories(face)
include_directories(serial/include)
include_directories(video)
include_directories(bat)

# name of executable file and path of source file
add_executable(main main.cpp face/face_api.cpp
               serial/src/serialport.cpp serial/src/CRC_Check.cpp
               video/RMVideoCapture.cpp bat/bat.cpp)
# opencv libraries
target_link_libraries(main  ${OpenCV_LIBS})
target_link_libraries(main -lpthread -lm)