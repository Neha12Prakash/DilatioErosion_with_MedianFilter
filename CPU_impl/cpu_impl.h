//
// Created by amey on 05.07.21.
//

#ifndef GPU_TEAM2_PROJECT_CPU_IMPL_H
#define GPU_TEAM2_PROJECT_CPU_IMPL_H

struct structuring_element{
    int* struct_kernel;
    int sizeX;
    int sizeY;
    int can_x;
    int can_y;
};

void binary_invert(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY);

void grayscale_invert(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY);

void image_threshold(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY, float threshold);

void median_filter(const std::vector<float> &h_input, std::size_t kernel_size, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY);

void add_salt_and_pepper(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY);

void dilate(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY, structuring_element struc_ele);

void erode(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY, structuring_element struc_ele);

#endif //GPU_TEAM2_PROJECT_CPU_IMPL_H
