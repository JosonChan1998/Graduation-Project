# Graduation Project
This is the vision program of the ping-pong transmitter.

## Environment
- Ubuntu 18.04
- OpenCV 4.5.1
- g++
- CMake
- USB Camera (Use Linux v4l2 drive)
- URAT device (USB2TTY)

## Build target
```bash
mkdir build
cd build
cmake ../
make
```

## Launch
```bash
pwd # ~/grad_project/build
./main
```
## Tune Parammter
You can see all parameter in `config.hpp`.While you turnning it, you need to rebuild the project.


