#ifndef __OPENCL_VERSION__

#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work

#endif

//////////// STRUCUTRES ///////////////
struct structuring_element{
    int* struct_kernel;
    int sizeX;
    int sizeY;
    int can_x;
    int can_y;
};
///////////////////////////////////////


int getIndexGlobal(size_t countX, size_t i, size_t j) {
    return j * countX + i;
}

float getValueGlobal(__global const float *a, size_t countX, size_t countY, size_t i, size_t j) {
    if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
        return 0;
    else
        return a[getIndexGlobal(countX, i, j)];
}


__kernel void erode(__global const float *d_input, __global float *d_output, __global int *struct_kernel, int sizeX, int sizeY) {
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    bool is_odd_x, is_odd_y;
    size_t candidate_x, candidate_y;

    size_t n = sizeX * sizeY;

    if((sizeX % 2) == 1) is_odd_x = true;
    else is_odd_x = false;

    if((sizeY % 2) == 1) is_odd_y = true;
    else is_odd_y = false;

    if(is_odd_x){
        candidate_x = ((size_t)(sizeX / 2));
    }
    else{
        candidate_x = ((size_t)(sizeX / 2) - 1);
    }

    if(is_odd_y){
        candidate_y = ((size_t)(sizeY / 2));
    }
    else{
        candidate_y = ((size_t)(sizeY / 2) - 1);
    }

    bool hit = false;
    for (int temp_i = i-candidate_x; temp_i <= i+candidate_x ; ++temp_i) {
        for (int temp_j = j - candidate_y; temp_j <= j + candidate_y; ++temp_j) {
            int temp_index_i = temp_i - (i-candidate_x);
            int temp_index_j = temp_j - (j-candidate_y);

            if(
                ((struct_kernel[temp_index_i+(sizeX*temp_index_j)]) == 1)
                &&
                (getValueGlobal(d_input, countX, countY, temp_i, temp_j) == 1)
                ){
            hit = true;

        }
        else{
            hit = false;
            break;
        }
        }
    if(hit){
        break;
    }
    }
    if(hit) d_output[getIndexGlobal(countX, i, j)] = 1;
    else d_output[getIndexGlobal(countX, i, j)] = 0;

}

const sampler_t erosion_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST ;
__kernel void erode_image(__read_only image2d_t d_input, __read_write image2d_t d_output, __global int *struct_kernel, int sizeX, int sizeY) {
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    bool is_odd_x, is_odd_y;
    size_t candidate_x, candidate_y;

    size_t n = sizeX * sizeY;

    if((sizeX % 2) == 1) is_odd_x = true;
    else is_odd_x = false;

    if((sizeY % 2) == 1) is_odd_y = true;
    else is_odd_y = false;

    if(is_odd_x){
        candidate_x = ((size_t)(sizeX / 2));
    }
    else{
        candidate_x = ((size_t)(sizeX / 2) - 1);
    }

    if(is_odd_y){
        candidate_y = ((size_t)(sizeY / 2));
    }
    else{
        candidate_y = ((size_t)(sizeY / 2) - 1);
    }

    bool hit = false;
    for (int temp_i = i-candidate_x; temp_i <= i+candidate_x ; ++temp_i) {
        for (int temp_j = j - candidate_y; temp_j <= j + candidate_y; ++temp_j) {
            int temp_index_i = temp_i - (i-candidate_x);
            int temp_index_j = temp_j - (j-candidate_y);

            if(
                    ((struct_kernel[temp_index_i+(sizeX*temp_index_j)]) == 1)
                    &&
                    (read_imagef(d_input, (int2){temp_i, temp_j}).x == 1)
            ){
                hit = true;
            }
            else{
                hit = false;
                break;
            }
        }
        if(hit) break;
    }
    if (hit){
        write_imagef(d_output, (int2){i, j}, (float4)(1,0,0,1));
    }
    else{
        write_imagef(d_output, (int2){i, j}, (float4)(0,0,0,1));
    }

}