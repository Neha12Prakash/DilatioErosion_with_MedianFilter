#ifndef __OPENCL_VERSION__

#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work

#endif

// Function to perform Selection Sort
void selectionSort(float arr[], int n)
{
    int i, j, min_idx;
    for (i = 0; i < n - 1; i++) {
        min_idx = i;
        for (j = i + 1; j < n; j++)
            if (arr[j] < arr[min_idx])
                min_idx = j;
        float temp;
        temp = arr[min_idx];
        arr[min_idx] = arr[i];
        arr[i] = temp;

    }
}

//TODO
int getIndexGlobal(size_t countX, size_t i, size_t j) {
    return j * countX + i;
}

float getValueGlobal(__global const float *a, size_t countX, size_t countY, size_t i, size_t j) {
    if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
        return 0;
    else
        return a[getIndexGlobal(countX, i, j)];
}

__kernel void medianKernel(__global const float *d_input, __global float *d_output) {
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    bool is_odd;
    int candidate;
    const int kernel_size = KERNEL_SIZE_MAIN;

    const int n = kernel_size*kernel_size;
    float array[9];

    if((kernel_size % 2) == 1) is_odd = true;
    else is_odd = false;

    if(is_odd){
        candidate = ((kernel_size / 2));
    }
    else{
        candidate = ((kernel_size / 2) - 1);
    }

    for (int temp_i = i-candidate; temp_i <= i+candidate ; ++temp_i) {
        for (int temp_j = j-candidate; temp_j <= j+candidate; ++temp_j) {
            int temp_index_i = temp_i - (i-candidate);
            int temp_index_j = temp_j - (j-candidate);

            array[temp_index_i + kernel_size*temp_index_j] = getValueGlobal(d_input, countX, countY, temp_i, temp_j);
        }
    }

    selectionSort(array, n);

    d_output[getIndexGlobal(countX, i, j)] = array[candidate + kernel_size*candidate];


}

const sampler_t median_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST ; 
__kernel void medianKernel_image(__read_only image2d_t d_input, __read_write image2d_t d_output) {
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    
    bool is_odd;
    int candidate;
    const int kernel_size = KERNEL_SIZE_MAIN;

    const int n = kernel_size*kernel_size;
    float array[9];

    if((kernel_size % 2) == 1) is_odd = true;
    else is_odd = false;

    if(is_odd){
        candidate = ((kernel_size / 2));
    }
    else{
        candidate = ((kernel_size / 2) - 1);
    }

    for (int temp_i = i-candidate; temp_i <= i+candidate ; ++temp_i) {
        for (int temp_j = j-candidate; temp_j <= j+candidate; ++temp_j) {
            int temp_index_i = temp_i - (i-candidate);
            int temp_index_j = temp_j - (j-candidate);

            array[temp_index_i + kernel_size*temp_index_j] = read_imagef(d_input, (int2){temp_i, temp_j}).x;
        }
    }

    selectionSort(array, n);

    write_imagef(d_output, (int2){i, j}, (float4)(array[candidate + kernel_size*candidate], 0, 0, 1));
}

/////////////////////// DEPRECATED ///////////////////////////////

__kernel void medianKernel_3(__global const float *d_input, __global float *d_output) {
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    bool is_odd;
    int candidate;
    const int kernel_size = 3;

    const int n = kernel_size*kernel_size;
    float array[9];

    if((kernel_size % 2) == 1) is_odd = true;
    else is_odd = false;

    if(is_odd){
        candidate = ((kernel_size / 2));
    }
    else{
        candidate = ((kernel_size / 2) - 1);
    }

    for (int temp_i = i-candidate; temp_i <= i+candidate ; ++temp_i) {
        for (int temp_j = j-candidate; temp_j <= j+candidate; ++temp_j) {
            int temp_index_i = temp_i - (i-candidate);
            int temp_index_j = temp_j - (j-candidate);

            array[temp_index_i + kernel_size*temp_index_j] = getValueGlobal(d_input, countX, countY, temp_i, temp_j);
        }
    }

    selectionSort(array, n);

    d_output[getIndexGlobal(countX, i, j)] = array[candidate + kernel_size*candidate];


}


__kernel void medianKernel_5(__global const float *d_input, __global float *d_output) {
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    bool is_odd;
    int candidate;
    const int kernel_size = 5;

    const int n = kernel_size*kernel_size;
    float array[25];

    if((kernel_size % 2) == 1) is_odd = true;
    else is_odd = false;

    if(is_odd){
        candidate = ((kernel_size / 2));
    }
    else{
        candidate = ((kernel_size / 2) - 1);
    }

    for (int temp_i = i-candidate; temp_i <= i+candidate ; ++temp_i) {
        for (int temp_j = j-candidate; temp_j <= j+candidate; ++temp_j) {
            int temp_index_i = temp_i - (i-candidate);
            int temp_index_j = temp_j - (j-candidate);

            array[temp_index_i + kernel_size*temp_index_j] = getValueGlobal(d_input, countX, countY, temp_i, temp_j);
        }
    }

    selectionSort(array, n);

    d_output[getIndexGlobal(countX, i, j)] = array[candidate + kernel_size*candidate];


}

const sampler_t median_sampler_3 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST ; 
__kernel void medianKernel_image_3(__read_only image2d_t d_input, __read_write image2d_t d_output) {
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    
    bool is_odd;
    int candidate;
    const int kernel_size = 3;

    const int n = kernel_size*kernel_size;
    float array[9];

    if((kernel_size % 2) == 1) is_odd = true;
    else is_odd = false;

    if(is_odd){
        candidate = ((kernel_size / 2));
    }
    else{
        candidate = ((kernel_size / 2) - 1);
    }

    for (int temp_i = i-candidate; temp_i <= i+candidate ; ++temp_i) {
        for (int temp_j = j-candidate; temp_j <= j+candidate; ++temp_j) {
            int temp_index_i = temp_i - (i-candidate);
            int temp_index_j = temp_j - (j-candidate);

            array[temp_index_i + kernel_size*temp_index_j] = read_imagef(d_input, (int2){temp_i, temp_j}).x;
        }
    }

    selectionSort(array, n);

    write_imagef(d_output, (int2){i, j}, (float4)(array[candidate + kernel_size*candidate], 0, 0, 1));
}

const sampler_t median_sampler_5 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST ; 
__kernel void medianKernel_image_5(__read_only image2d_t d_input, __read_write image2d_t d_output) {
    size_t countX = get_global_size(0);
    size_t countY = get_global_size(1);
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    
    bool is_odd;
    int candidate;
    const int kernel_size = 5;

    const int n = kernel_size*kernel_size;
    float array[25];

    if((kernel_size % 2) == 1) is_odd = true;
    else is_odd = false;

    if(is_odd){
        candidate = ((kernel_size / 2));
    }
    else{
        candidate = ((kernel_size / 2) - 1);
    }

    for (int temp_i = i-candidate; temp_i <= i+candidate ; ++temp_i) {
        for (int temp_j = j-candidate; temp_j <= j+candidate; ++temp_j) {
            int temp_index_i = temp_i - (i-candidate);
            int temp_index_j = temp_j - (j-candidate);

            array[temp_index_i + kernel_size*temp_index_j] = read_imagef(d_input, (int2){temp_i, temp_j}).x;
        }
    }

    selectionSort(array, n);

    write_imagef(d_output, (int2){i, j}, (float4)(array[candidate + kernel_size*candidate], 0, 0, 1));
}