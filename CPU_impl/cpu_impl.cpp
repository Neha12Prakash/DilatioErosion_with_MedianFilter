//
// Created by amey on 05.07.21.
//

// Includes
#include <stdio.h>

#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <cmath>
#include <iostream>


#include "cpu_impl.h"
// Structures




// Private Functions

int getIndexGlobal(std::size_t countX, int i, int j) {
    return j * countX + i;
}

// Read value from global array a, return 0 if outside image
float getValueGlobal(const std::vector<float> &a, std::size_t countX, std::size_t countY, int i, int j) {
    if (i < 0 || (size_t) i >= countX || j < 0 || (size_t) j >= countY)
        return 0;
    else
        return a[getIndexGlobal(countX, i, j)];
}


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



/*
 * binary_invert() :                Converts a bitmap image into a binary inverted image
 * grayscale_invert() :             Converts a grayscale image into it's invert
 * convert_pgm_to_bitmap() :        Converts a PGM to a Bitmap
 * */


void binary_invert(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY){
    for (int i = 0; i < (int) countX; i++) {
        for (int j = 0; j < (int) countY; j++) {
            if(getValueGlobal(h_input, countX, countY, i, j) == 1){
                h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
            }
            else{
                h_outputCpu[getIndexGlobal(countX, i, j)] = 1;
            }
        }
    }
}

void grayscale_invert(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY){
    for (int i = 0; i < (int) countX; i++) {
        for (int j = 0; j < (int) countY; j++) {
                h_outputCpu[getIndexGlobal(countX, i, j)] = 1 - getValueGlobal(h_input, countX, countY, i, j);
        }
    }
}

void image_threshold(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY, float threshold){
    for (int i = 0; i < (int) countX; i++) {
        for (int j = 0; j < (int) countY; j++) {
            float val = getValueGlobal(h_input, countX, countY, i, j);
            if(val <= threshold){
                h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
            }
            else{
                h_outputCpu[getIndexGlobal(countX, i, j)] = 1;
            }
        }
    }
}

void median_filter(const std::vector<float> &h_input, std::size_t kernel_size, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY){
    bool is_odd;
    int candidate;

    int n = kernel_size * kernel_size;
    float array[n];

    if((kernel_size % 2) == 1) is_odd = true;
    else is_odd = false;

    if(is_odd){
        candidate = ((std::size_t)(kernel_size / 2));
    }
    else{
        candidate = ((std::size_t)(kernel_size / 2) - 1);
    }
    std::cout << "The candidate pixel is (" << candidate << ", " << candidate << ")" << std::endl;


    for(int i = candidate; i < countX - candidate; ++i){
        for (int j = candidate; j < countY - candidate; ++j) {
            for (int temp_i = i-candidate; temp_i <= i+candidate ; ++temp_i) {
                for (int temp_j = j-candidate; temp_j <= j+candidate; ++temp_j) {
                    int temp_index_i = temp_i - (i-candidate);
                    int temp_index_j = temp_j - (j-candidate);

                    array[temp_index_i + kernel_size*temp_index_j] = getValueGlobal(h_input, countX, countY, temp_i, temp_j);
                }
            }

            selectionSort(array, n);

            h_outputCpu[getIndexGlobal(countX, i, j)] = array[candidate + kernel_size*candidate];
        }
    }

}


void add_salt_and_pepper(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY){

    for (int i = 0; i < (int) countX; i++) {
        for (int j = 0; j < (int) countY; j++) {
            h_outputCpu[getIndexGlobal(countX, i, j)] = getValueGlobal(h_input, countX, countY, i, j);
        }
    }

    for (int i = 0; i < 10000; ++i) {
        h_outputCpu[getIndexGlobal(countX, (int )(rand() % countX), (int)(rand() % countY)) ]= 1;
    }
    for (int i = 0; i < 10000; ++i) {
        h_outputCpu[getIndexGlobal(countX, (int )(rand() % countX), (int)(rand() % countY)) ]= 0;
    }
}

void dilate(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY, structuring_element struc_ele){

    bool is_odd_x, is_odd_y;
    int candidate_x, candidate_y;

    int n = struc_ele.sizeX * struc_ele.sizeY;

    if((struc_ele.sizeX % 2) == 1) is_odd_x = true;
    else is_odd_x = false;

    if((struc_ele.sizeY % 2) == 1) is_odd_y = true;
    else is_odd_y = false;

    if(is_odd_x){
        candidate_x = ((std::size_t)(struc_ele.sizeX / 2));
    }
    else{
        candidate_x = ((std::size_t)(struc_ele.sizeX / 2) - 1);
    }

    if(is_odd_y){
        candidate_y = ((std::size_t)(struc_ele.sizeY / 2));
    }
    else{
        candidate_y = ((std::size_t)(struc_ele.sizeY / 2) - 1);
    }


    for(int i = candidate_x; i < countX - candidate_x; ++i) {
        for (int j = candidate_y; j < countY - candidate_y; ++j) {
            bool hit = false;
            for (int temp_i = i-candidate_x; temp_i <= i+candidate_x ; ++temp_i) {
                for (int temp_j = j - candidate_y; temp_j <= j + candidate_y; ++temp_j) {
                    int temp_index_i = temp_i - (i-candidate_x);
                    int temp_index_j = temp_j - (j-candidate_y);

                    if(
                            ((struc_ele.struct_kernel[temp_index_i+(struc_ele.sizeX*temp_index_j)]) == 1)
                            &&
                            (getValueGlobal(h_input, countX, countY, temp_i, temp_j) == 1)
                    ){
                        hit = true;
                        break;
                    }
                    else{
                        hit = false;
                    }
                }
                if(hit) break;
            }
            if (hit){
                h_outputCpu[getIndexGlobal(countX, i, j)] = 1;
            }
            else{
                h_outputCpu[getIndexGlobal(countX, i, j)] = getValueGlobal(h_input, countX, countY, i, j);
            }
        }
    }

}


void erode(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY, structuring_element struc_ele){

    bool is_odd_x, is_odd_y;
    int candidate_x, candidate_y;

    int n = struc_ele.sizeX * struc_ele.sizeY;

    if((struc_ele.sizeX % 2) == 1) is_odd_x = true;
    else is_odd_x = false;

    if((struc_ele.sizeY % 2) == 1) is_odd_y = true;
    else is_odd_y = false;

    if(is_odd_x){
        candidate_x = ((std::size_t)(struc_ele.sizeX / 2));
    }
    else{
        candidate_x = ((std::size_t)(struc_ele.sizeX / 2) - 1);
    }

    if(is_odd_y){
        candidate_y = ((std::size_t)(struc_ele.sizeY / 2));
    }
    else{
        candidate_y = ((std::size_t)(struc_ele.sizeY / 2) - 1);
    }


    for(int i = candidate_x; i < countX - candidate_x; ++i) {
        for (int j = candidate_y; j < countY - candidate_y; ++j) {
            bool hit = false;
            for (int temp_i = i-candidate_x; temp_i <= i+candidate_x ; ++temp_i) {
                for (int temp_j = j - candidate_y; temp_j <= j + candidate_y; ++temp_j) {
                    int temp_index_i = temp_i - (i-candidate_x);
                    int temp_index_j = temp_j - (j-candidate_y);

                    if(
                            ((struc_ele.struct_kernel[temp_index_i+(struc_ele.sizeX*temp_index_j)]) == 1)
                            &&
                            (getValueGlobal(h_input, countX, countY, temp_i, temp_j) == 1)
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
                if(hit) h_outputCpu[getIndexGlobal(countX, i, j)] = 1;
                else h_outputCpu[getIndexGlobal(countX, i, j)] = 0;
            }


        }
    }







/**
 * References
 * */

void sobelHost(const std::vector<float> &h_input, std::vector<float> &h_outputCpu, std::size_t countX, std::size_t countY) {
    for (int i = 0; i < (int) countX; i++) {
        for (int j = 0; j < (int) countY; j++) {
            float Gx = getValueGlobal(h_input, countX, countY, i - 1, j - 1) +
                       2 * getValueGlobal(h_input, countX, countY, i - 1, j) +
                       getValueGlobal(h_input, countX, countY, i - 1, j + 1)
                       - getValueGlobal(h_input, countX, countY, i + 1, j - 1) -
                       2 * getValueGlobal(h_input, countX, countY, i + 1, j) -
                       getValueGlobal(h_input, countX, countY, i + 1, j + 1);
            float Gy = getValueGlobal(h_input, countX, countY, i - 1, j - 1) +
                       2 * getValueGlobal(h_input, countX, countY, i, j - 1) +
                       getValueGlobal(h_input, countX, countY, i + 1, j - 1)
                       - getValueGlobal(h_input, countX, countY, i - 1, j + 1) -
                       2 * getValueGlobal(h_input, countX, countY, i, j + 1) -
                       getValueGlobal(h_input, countX, countY, i + 1, j + 1);
            h_outputCpu[getIndexGlobal(countX, i, j)] = sqrt(Gx * Gx + Gy * Gy);
        }
    }
}
