//
// Created by amey on 05.07.21.
//

// includes
#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include <boost/lexical_cast.hpp>

// Include the CPU Implementation
#include "CPU_impl/cpu_impl.h"

///////////// Declarations

//std::string input_image_path = "../Images/Inputs/Binary_Image/image_1.pgm";
//std::string input_image_path = "../Images/Valve.pgm";
//std::string output_path = "../Images/Outputs/";


///////////// Local Functions/ Methods

/**
 * getCountSize() :         Returns all the countX, countY, count and size based on image width and height
 * */

auto getCount_Size(
        std::size_t wgSizeX,
        std::size_t wgSizeY,
        std::size_t inWidth,
        std::size_t inHeight
        ){
    std::size_t cntX, cntY, cnt, sze;
    cntX = wgSizeX * std::size_t(inWidth / wgSizeX);
    cntY = wgSizeY * std::size_t(inHeight / wgSizeY);
    cnt = cntX * cntY;
    sze = cnt * sizeof(float);

    struct result {std::size_t countX; std::size_t countY; std::size_t count; std::size_t size;};
    return result {cntX, cntY, cnt, sze};

}

cl::Program buildProgram(cl::Context context, std::vector<cl::Device> devices, std::string filepath){
    cl::Program program = OpenCL::loadProgramSource(context, filepath);
    OpenCL::buildProgram(program, devices);
    return program;
}


int main(int argc, char **argv) {
    /**
     * Define the command line arguments here
     * */

    std::string input_image_path = argv[1];
    std::string output_path = argv[2];


    // Create a context
    //cl::Context context(CL_DEVICE_TYPE_GPU);
    std::vector <cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        std::cerr << "No platforms found" << std::endl;
        return 1;
    }
    int platformId = 0;
    for (size_t i = 0; i < platforms.size(); i++) {
        if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
            platformId = i;
            break;
        }
    }
    cl_context_properties prop[4] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[platformId](), 0, 0};
    std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '"
              << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
    cl::Context context(CL_DEVICE_TYPE_GPU, prop);

    // Get the first device of the context
    std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
    cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
    std::vector <cl::Device> devices;
    devices.push_back(device);
    OpenCL::printDeviceInfo(std::cout, device);
    std::cout << "Number of Cores: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    std::cout << "Number of WorkGroups (max): " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
    std::cout << "Number of WorkItem Dimensions (max): " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;

    // Create a command queue
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Declare some values
    std::size_t wgSizeX = 32;                          // Number of work items per work group in X direction
    std::size_t wgSizeY = 32;
    /**
     * Read the test image here
     * */
    std::vector<float> inputData;
    std::size_t inputWidth, inputHeight;
    Core::readImagePGM(input_image_path, inputData, inputWidth, inputHeight);
    auto[countX, countY, count, size] = getCount_Size(
            wgSizeX,
            wgSizeY,
            inputWidth,
            inputHeight
    );

    // Size for the median kernel
    int median_kernel_size = 3;


    // Allocate space for output data from CPU and GPU on the host
    std::vector<float> h_input(count);
    std::vector<float> temp(count);
    std::vector<float> h_outputCpu(count);
    std::vector<float> h_outputGpu(count);

    // Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
    memset(h_input.data(), 255, size);
    memset(temp.data(), 255, size);
    memset(h_outputCpu.data(), 255, size);
    memset(h_outputGpu.data(), 255, size);

    {

        for (size_t j = 0; j < countY; j++) {
            for (size_t i = 0; i < countX; i++) {
                h_input[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
            }
        }
    }

    //////////////// Do calculation on the host side : declare the structuring element
    int struc_ele_width = 5;
    int struc_ele_height = 5;
    int struc_element_count = struc_ele_width*struc_ele_height;
    std::vector<int> dilation_element(5*5);
    memset(dilation_element.data(), 0, struc_element_count*sizeof(int ));
//    dilation_element = {
//            0, 0, 1, 1, 1, 0, 0,
//            0, 0, 1, 1, 1, 0, 0,
//            1, 1, 1, 1, 1, 1, 1,
//            1, 1, 1, 1, 1, 1, 1,
//            1, 1, 1, 1, 1, 1, 1,
//            0, 0, 1, 1, 1, 0, 0,
//            0, 0, 1, 1, 1, 0, 0};

    dilation_element = {
            0, 0, 1, 0, 0,
            0, 1, 1, 1, 0,
            1, 1, 1, 1, 1,
            0, 1, 1, 1, 0,
            0, 0, 1, 0, 0};

//int dilation_element[3][3] = {
//        {0, 1, 0},
//        {1, 1, 1},
//        {0, 1, 0}
//};
    structuring_element a_struc_ele = {
            dilation_element.data(),
            struc_ele_width,
            struc_ele_height,
            3,                          //  Candidate pixel coordinates
            3                           //  Candidate pixel coordinates
    };

    ////////////////// Support Operations /////////////////////////

//    add_salt_and_pepper(h_input, h_outputCpu, countX, countY);
//    image_threshold(h_input, h_outputCpu, countX, countY, 0.5);
    ////////////////////////////////////////////////

    Core::TimeSpan t1 = Core::getCurrentTime();
    median_filter(h_input, median_kernel_size, h_outputCpu, countX, countY);
    Core::TimeSpan cpuTime_median = Core::getCurrentTime() - t1;
    Core::writeImagePGM(output_path+"output_median_cpu.pgm", h_outputCpu, countX, countY);

    //////////// Perform Dilation ////////////////////////////////////
    Core::TimeSpan t2 = Core::getCurrentTime();
    dilate(h_input, h_outputCpu, countX, countY, a_struc_ele);
    Core::TimeSpan cpuTime_dilation = Core::getCurrentTime() - t2;
    Core::writeImagePGM(output_path+"output_dilation_cpu.pgm", h_outputCpu, countX, countY);

    //////////// Perform Erosion ////////////////////////////////////
    Core::TimeSpan t3 = Core::getCurrentTime();
    erode(h_input, h_outputCpu, countX, countY, a_struc_ele);
    Core::TimeSpan cpuTime_erosion = Core::getCurrentTime() - t3;
    Core::writeImagePGM(output_path+"output_erosion_cpu.pgm", h_outputCpu, countX, countY);

    //////////// Perform Opening ////////////////////////////////////
    Core::TimeSpan t4 = Core::getCurrentTime();
    dilate(h_input, temp, countX, countY, a_struc_ele);
    erode(temp, h_outputCpu, countX, countY, a_struc_ele);
    Core::TimeSpan cpuTime_opening = Core::getCurrentTime() - t4;
    Core::writeImagePGM(output_path+"output_opening_cpu.pgm", h_outputCpu, countX, countY);

    //////////// Perform Closing ///////////////////////////////////
    Core::TimeSpan t5 = Core::getCurrentTime();
    erode(h_input, temp, countX, countY, a_struc_ele);
    dilate(temp, h_outputCpu, countX, countY, a_struc_ele);
    Core::TimeSpan cpuTime_closing = Core::getCurrentTime() - t5;
    Core::writeImagePGM(output_path+"output_closing_cpu.pgm", h_outputCpu, countX, countY);





    //////// Load Images in the device ////////////////////////////////

    auto d_input_image = cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
    auto d_output_image = cl::Image2D(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);

    cl::size_t<3> origin;
    origin[0] = origin[1] = origin[2] = 0;
    cl::size_t<3> region;
    region[0] = countX;
    region[1] = countY;
    region[2] = 1;

    cl::Event imageUpload;                              // image upload event

    queue.enqueueWriteImage(
            d_input_image,
            true,
            origin,
            region,
            countX * sizeof(float),
            0,
            h_input.data(),
            NULL,
            NULL
    );


    /////////// Load Buffers ///////////////////
    cl::Buffer d_input_buffer(context, CL_MEM_READ_WRITE, size);
    cl::Buffer d_output_buffer(context, CL_MEM_READ_WRITE, size);
    cl::Buffer temp_buffer(context, CL_MEM_READ_WRITE, size);
    cl::Buffer d_dilation_kernel(context, CL_MEM_READ_WRITE, struc_element_count*sizeof(int ));


    queue.enqueueWriteBuffer(d_input_buffer, true, 0, size, h_input.data());
    queue.enqueueWriteBuffer(d_dilation_kernel, true, 0, struc_element_count*sizeof(int ), dilation_element.data());

    /**
     * Load All the CL programs that you design here:
     * */

//    cl::Program median_program = buildProgram(context, devices, "../src/median.cl");
    cl::Program median_program = OpenCL::loadProgramSource(context, "../src/median.cl");
    OpenCL::buildProgram(
            median_program,
            devices, "-DKERNEL_SIZE_MAIN=" + boost::lexical_cast<std::string>(median_kernel_size)         // SPECIFY KERNEL SIZE
    );
    cl::Kernel median_kernel_3(median_program, "medianKernel");
    cl::Kernel median_kernel_image_3(median_program, "medianKernel_image");

//    cl::Program dilation_program = buildProgram(context, devices, "../src/dilation.cl");
    cl::Program dilation_program = OpenCL::loadProgramSource(context, "../src/dilation.cl");
    OpenCL::buildProgram(
            dilation_program,
            devices, "-DIMAGE_SIZE_X=" + boost::lexical_cast<std::string>(countX) + " -DIMAGE_SIZE_Y=" + boost::lexical_cast<std::string>(countY)
    );

    cl::Kernel dilation_kernel(dilation_program, "dilate");
    cl::Kernel dilation_kernel_image(dilation_program, "dilate_image");

    cl::Program erosion_program = buildProgram(context, devices, "../src/erosion.cl");
    cl::Kernel erosion_kernel(erosion_program, "erode");
    cl::Kernel erosion_kernel_image(erosion_program, "erode_image");


    cl::Program o_c_program = OpenCL::loadProgramSource(context, "../src/opening_closing.cl");
    OpenCL::buildProgram(
            o_c_program,
            devices, "-DIMAGE_SIZE_X=" + boost::lexical_cast<std::string>(countX) + " -DIMAGE_SIZE_Y=" + boost::lexical_cast<std::string>(countY)
    );
    cl::Kernel opening_kernel(o_c_program, "opening");
    /////////////////// Run the Kernel ///////////////////////


    ////////////////// Median Kernel : Bufferized ////////////
    cl::Event median_buffer_execution;
    median_kernel_3.setArg<cl::Buffer>(0, d_input_buffer);
    median_kernel_3.setArg<cl::Buffer>(1, d_output_buffer);

    queue.enqueueNDRangeKernel(median_kernel_3,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &median_buffer_execution);

    queue.enqueueReadBuffer(d_output_buffer, true, 0, size, h_outputGpu.data(), NULL, NULL);

    //////// Store GPU output image : Bufferized Median Kernel ///////////////////////////////////
    Core::writeImagePGM(output_path+"output_median_gpu_bufferized.pgm", h_outputGpu, countX, countY);

    std::cout << "INFO:\t MEDIAN FILTER (Bufferized):\tSpeedup = "<< (cpuTime_median.getSeconds() / OpenCL::getElapsedTime(median_buffer_execution).getSeconds()) << std::endl;


    //////// Median Kernel : with Image2D //////////////
    cl::Event median_image_execution;
    median_kernel_image_3.setArg<cl::Image2D>(0, d_input_image);
    median_kernel_image_3.setArg<cl::Image2D>(1, d_output_image);

    queue.enqueueNDRangeKernel(median_kernel_image_3,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &median_image_execution);

    queue.enqueueReadImage(d_output_image,
                           true,
                           origin,
                           region,
                           countX * sizeof(float),
                           0,
                           h_outputGpu.data(),
                           nullptr,
                           NULL
    );
    //////// Store GPU output image : Image Median Kernel ///////////////////////////////////
    Core::writeImagePGM(output_path+"output_median_gpu_image.pgm", h_outputGpu, countX, countY);
    std::cout << "INFO:\t MEDIAN FILTER (Image2D):\tSpeedup = "<< (cpuTime_median.getSeconds() / OpenCL::getElapsedTime(median_image_execution).getSeconds()) << std::endl;


    ////////// Dilation : Bufferized /////////////////////////////////////////
    cl::Event dilation_execution;
    dilation_kernel.setArg<cl::Buffer>(0, d_input_buffer);
    dilation_kernel.setArg<cl::Buffer>(1, d_output_buffer);
    dilation_kernel.setArg<cl::Buffer>(2, d_dilation_kernel);
    dilation_kernel.setArg<int>(3, struc_ele_width);
    dilation_kernel.setArg<int>(4, struc_ele_height);

    queue.enqueueNDRangeKernel(dilation_kernel,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &dilation_execution);

    queue.enqueueReadBuffer(d_output_buffer, true, 0, size, h_outputGpu.data(), NULL, NULL);

    //////// Store GPU output image : Bufferized Dilation ///////////////////////////////////
    Core::writeImagePGM(output_path+"output_dilation_bufferized_gpu.pgm", h_outputGpu, countX, countY);
    std::cout << "INFO:\t Dilation (Buffer):\tSpeedup = "<< (cpuTime_dilation.getSeconds() / OpenCL::getElapsedTime(dilation_execution).getSeconds()) << std::endl;

    ///////////////// Dilation : Image2D ///////////////////
    cl::Event dilation_image_execution;
    dilation_kernel_image.setArg<cl::Image2D>(0, d_input_image);
    dilation_kernel_image.setArg<cl::Image2D>(1, d_output_image);
    dilation_kernel_image.setArg<cl::Buffer>(2, d_dilation_kernel);
    dilation_kernel_image.setArg<int>(3, struc_ele_width);
    dilation_kernel_image.setArg<int>(4, struc_ele_height);


    queue.enqueueNDRangeKernel(dilation_kernel_image,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &dilation_image_execution);

    queue.enqueueReadImage(d_output_image,
                           true,
                           origin,
                           region,
                           countX * sizeof(float),
                           0,
                           h_outputGpu.data(),
                           nullptr,
                           NULL
    );
    //////// Store GPU output image : Image Dilation Kernel ///////////////////////////////////
    Core::writeImagePGM(output_path+"output_dilation_gpu_image.pgm", h_outputGpu, countX, countY);
    std::cout << "INFO:\t Dilation (Image2D):\tSpeedup = "<< (cpuTime_dilation.getSeconds() / OpenCL::getElapsedTime(dilation_image_execution).getSeconds()) << std::endl;



    ////////// Erosion : Bufferized /////////////////////////////////////////
    cl::Event erosion_execution;
    erosion_kernel.setArg<cl::Buffer>(0, d_input_buffer);
    erosion_kernel.setArg<cl::Buffer>(1, d_output_buffer);
    erosion_kernel.setArg<cl::Buffer>(2, d_dilation_kernel);
    erosion_kernel.setArg<int>(3, struc_ele_width);
    erosion_kernel.setArg<int>(4, struc_ele_height);

    queue.enqueueNDRangeKernel(erosion_kernel,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &erosion_execution);

    queue.enqueueReadBuffer(d_output_buffer, true, 0, size, h_outputGpu.data(), NULL, NULL);

    //////// Store GPU output image : Bufferized Erosino ///////////////////////////////////
    Core::writeImagePGM(output_path+"output_erosion_bufferized_gpu.pgm", h_outputGpu, countX, countY);
    std::cout << "INFO:\t Erosion (Buffer):\tSpeedup = "<< (cpuTime_erosion.getSeconds() / OpenCL::getElapsedTime(erosion_execution).getSeconds()) << std::endl;

    ///////////////// Erosion : Image2D ///////////////////
    cl::Event erosion_image_execution;
    erosion_kernel_image.setArg<cl::Image2D>(0, d_input_image);
    erosion_kernel_image.setArg<cl::Image2D>(1, d_output_image);
    erosion_kernel_image.setArg<cl::Buffer>(2, d_dilation_kernel);
    erosion_kernel_image.setArg<int>(3, struc_ele_width);
    erosion_kernel_image.setArg<int>(4, struc_ele_height);


    queue.enqueueNDRangeKernel(erosion_kernel_image,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &erosion_image_execution);

    queue.enqueueReadImage(d_output_image,
                           true,
                           origin,
                           region,
                           countX * sizeof(float),
                           0,
                           h_outputGpu.data(),
                           nullptr,
                           NULL
    );
    //////// Store GPU output image : Image Erosion Kernel ///////////////////////////////////
    Core::writeImagePGM(output_path+"output_erosion_gpu_image.pgm", h_outputGpu, countX, countY);
    std::cout << "INFO:\t Erosion (Image2D):\tSpeedup = "<< (cpuTime_erosion.getSeconds() / OpenCL::getElapsedTime(erosion_image_execution).getSeconds()) << std::endl;

    //////// Opening /////////////////////////////////////////////////////////////////////////
    cl::Event opening_execution;
    dilation_kernel.setArg<cl::Buffer>(0, d_input_buffer);
    dilation_kernel.setArg<cl::Buffer>(1, temp_buffer);
    dilation_kernel.setArg<cl::Buffer>(2, d_dilation_kernel);
    dilation_kernel.setArg<int>(3, struc_ele_width);
    dilation_kernel.setArg<int>(4, struc_ele_height);

    queue.enqueueNDRangeKernel(dilation_kernel,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &opening_execution);

    erosion_kernel.setArg<cl::Buffer>(0, temp_buffer);
    erosion_kernel.setArg<cl::Buffer>(1, d_output_buffer);
    erosion_kernel.setArg<cl::Buffer>(2, d_dilation_kernel);
    erosion_kernel.setArg<int>(3, struc_ele_width);
    erosion_kernel.setArg<int>(4, struc_ele_height);

    queue.enqueueNDRangeKernel(erosion_kernel,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &opening_execution);


    queue.enqueueReadBuffer(d_output_buffer, true, 0, size, h_outputGpu.data(), NULL, NULL);
    Core::writeImagePGM(output_path+"opening_gpu_image.pgm", h_outputGpu, countX, countY);
    std::cout << "INFO:\t Opening :\tSpeedup = "<< (cpuTime_opening.getSeconds() / OpenCL::getElapsedTime(opening_execution).getSeconds()) << std::endl;

    //////////////////////////// CLOSING ////////////////////////////////////
    cl::Event closing_execution;
    erosion_kernel.setArg<cl::Buffer>(0, d_input_buffer);
    erosion_kernel.setArg<cl::Buffer>(1, temp_buffer);
    erosion_kernel.setArg<cl::Buffer>(2, d_dilation_kernel);
    erosion_kernel.setArg<int>(3, struc_ele_width);
    erosion_kernel.setArg<int>(4, struc_ele_height);

    queue.enqueueNDRangeKernel(erosion_kernel,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &closing_execution);

    dilation_kernel.setArg<cl::Buffer>(0, temp_buffer);
    dilation_kernel.setArg<cl::Buffer>(1, d_output_buffer);
    dilation_kernel.setArg<cl::Buffer>(2, d_dilation_kernel);
    dilation_kernel.setArg<int>(3, struc_ele_width);
    dilation_kernel.setArg<int>(4, struc_ele_height);

    queue.enqueueNDRangeKernel(dilation_kernel,
                               cl::NullRange,
                               cl::NDRange(countX, countY),
                               cl::NDRange(wgSizeX, wgSizeY),
                               NULL,
                               &closing_execution);

    queue.enqueueReadBuffer(d_output_buffer, true, 0, size, h_outputGpu.data(), NULL, NULL);
    Core::writeImagePGM(output_path+"closing_gpu_image.pgm", h_outputGpu, countX, countY);
    std::cout << "INFO:\t Closing :\tSpeedup = "<< (cpuTime_closing.getSeconds() / OpenCL::getElapsedTime(closing_execution).getSeconds()) << std::endl;



}
