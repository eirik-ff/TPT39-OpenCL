#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <chrono>

#include "helpers.h"

using namespace cv;
using namespace std;

#define STRING_BUFFER_LEN 1024

#define SHOW 1
#define GPU_GAUSSIAN 0
#define GPU_SOBEL 0
#define GPU_AVERAGE 1
#define GPU_THRESHOLD 1


/* docs and notes

OpenCV Mat object is stored in row-major order, see https://docs.opencv.org/2.4/modules/core/doc/basic_structures.html#mat

*/


int main(int argc, char** argv)
{
    // defined as variables to be able to send them to kernels
    const int THRESH_VAL = 80;
    const int THRESH_MAXVAL = 255;

    // setup opencl
    char char_buffer[STRING_BUFFER_LEN];
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_context_properties context_properties[] =
    { 
        CL_CONTEXT_PLATFORM, 0,
        CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
        CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
        0
    };
    cl_command_queue queue;
    cl_program program;
    cl_kernel convolve_kernel, threshold_kernel, average_kernel;
    int status, success;

    // initialize OpenCl 
    clGetPlatformIDs(1, &platform, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

    context_properties[1] = (cl_context_properties)platform;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(context_properties, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    // build kernels
    unsigned char **source;
    
    // convolve kernel
    source = read_file("convolve.cl");
    program = clCreateProgramWithSource(context, 1, (const char **)source, NULL, NULL);
    if (program == NULL)
    {
        printf("Program creation failed\n");
        return EXIT_FAILURE;
    }	
    success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(success != CL_SUCCESS) print_clbuild_errors(program,device);
    convolve_kernel = clCreateKernel(program, "convolve", NULL);

    // threshold kernel
    source = read_file("threshold.cl");
    program = clCreateProgramWithSource(context, 1, (const char **)source, NULL, NULL);
    if (program == NULL)
    {
        printf("Program creation failed\n");
        return EXIT_FAILURE;
    }	
    success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(success != CL_SUCCESS) print_clbuild_errors(program,device);
    threshold_kernel = clCreateKernel(program, "threshold", NULL);


    // average kernel
    source = read_file("average.cl");
    program = clCreateProgramWithSource(context, 1, (const char **)source, NULL, NULL);
    if (program == NULL)
    {
        printf("Program creation failed\n");
        return EXIT_FAILURE;
    }	
    success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(success != CL_SUCCESS) print_clbuild_errors(program,device);
    average_kernel = clCreateKernel(program, "average", NULL);


    // load video
    VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    Size size = Size( (int)camera.get(CV_CAP_PROP_FRAME_WIDTH), (int)camera.get(CV_CAP_PROP_FRAME_HEIGHT) );
    cout << "SIZE: " << size << endl;

    // Open the output
    const string output_filename = "./output.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    VideoWriter outputVideo;
    outputVideo.open(output_filename, ex, 25, size, true);
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << output_filename << endl;
        return -1;
    }

    int tot_ms = 0;
    int count = 0;
    const char *window_name = "filter";   // Name shown in the GUI window.

#if SHOW
    namedWindow(window_name); // Resizable window, might not work on Windows.
    waitKey(1);
#endif

    size_t frame_size_px = size.width * size.height;
    size_t frame_size_bytes = frame_size_px * sizeof(unsigned char);
    size_t gaussian_kern_size = 3;
    size_t sobel_kern_size = 3;

    cl_mem grayframe_cl, edge_x_cl, edge_y_cl, edge_cl, gaussian_cl, sobel_x_cl, sobel_y_cl;

    grayframe_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, frame_size_bytes, NULL, &status);
    checkError(status, "Failed to allocate grayframe buffer");

    edge_x_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, frame_size_bytes, NULL, &status);
    checkError(status, "Failed to allocate edge_x buffer");

    edge_y_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, frame_size_bytes, NULL, &status);
    checkError(status, "Failed to allocate edge_y buffer");

    edge_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, frame_size_bytes, NULL, &status);
    checkError(status, "Failed to allocate edge buffer");

    gaussian_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, gaussian_kern_size * gaussian_kern_size * sizeof(float), NULL, &status);
    checkError(status, "Failed to allocate gaussian kernel buffer");

    sobel_x_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sobel_kern_size * sobel_kern_size * sizeof(float), NULL, &status);
    checkError(status, "Failed to allocal sobel x kernel buffer");

    sobel_y_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sobel_kern_size * sobel_kern_size * sizeof(float), NULL, &status);
    checkError(status, "Failed to allocal sobel y kernel buffer");


    // set threshold kernel args
    status = clSetKernelArg(threshold_kernel, 0, sizeof(cl_mem), &edge_cl);
    checkError(status, "Failed to set img param in threshold kernel");
    status = clSetKernelArg(threshold_kernel, 1, sizeof(int), &THRESH_VAL);
    checkError(status, "Failed to set thresh param in threshold kernel");
    status = clSetKernelArg(threshold_kernel, 2, sizeof(int), &THRESH_MAXVAL);
    checkError(status, "Failed to set maxval param in threshold kernel");


    // set average kernel args
    status = clSetKernelArg(average_kernel, 0, sizeof(cl_mem), &edge_x_cl);
    checkError(status, "Failed to set in1 param in average kernel");
    
    status = clSetKernelArg(average_kernel, 1, sizeof(cl_mem), &edge_y_cl);
    checkError(status, "Failed to set in2 param in average kernel");

    status = clSetKernelArg(average_kernel, 2, sizeof(cl_mem), &edge_cl);
    checkError(status, "Failed to set out param in average kernel");


    unsigned char *grayframe_ptr = NULL, *edge_x_ptr = NULL, *edge_y_ptr = NULL, *edge_ptr = NULL;

    grayframe_ptr = (unsigned char *)clEnqueueMapBuffer(queue, grayframe_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, frame_size_bytes, 0, NULL, NULL, &status);
    checkError(status, "Failed to map grayframe buffer to pointer");

    edge_x_ptr = (unsigned char *)clEnqueueMapBuffer(queue, edge_x_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, frame_size_bytes, 0, NULL, NULL, &status);
    checkError(status, "Failed to map edge_x buffer to pointer");

    edge_y_ptr = (unsigned char *)clEnqueueMapBuffer(queue, edge_y_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, frame_size_bytes, 0, NULL, NULL, &status);
    checkError(status, "Failed to map edge_y buffer to pointer");

    edge_ptr = (unsigned char *)clEnqueueMapBuffer(queue, edge_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, frame_size_bytes, 0, NULL, NULL, &status);
    checkError(status, "Failed to map edge buffer to pointer");

    Mat grayframe(size, CV_8U, grayframe_ptr);
    Mat edge_x(size, CV_8U, edge_x_ptr);
    Mat edge_y(size, CV_8U, edge_y_ptr);
    Mat edge(size, CV_8U, edge_ptr);

    // set gaussian convolution kernel
    float *gaussian_ptr = (float *)clEnqueueMapBuffer(queue, gaussian_cl, CL_TRUE, CL_MAP_WRITE, 0, gaussian_kern_size * gaussian_kern_size, 0, NULL, NULL, &status);
    checkError(status, "Failed to map gaussian kernel buffer to pointer");
    // kernel values copied from 3x3 from wikipedia: https://en.wikipedia.org/wiki/Kernel_(image_processing)
    *gaussian_ptr = { 1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16 };
    clEnqueueUnmapMemObject(queue, gaussian_cl, gaussian_ptr, 0, NULL, NULL);

    // set sobel x convolution kernel
    float *sobel_x_ptr = (float *)clEnqueueMapBuffer(queue, sobel_x_cl, CL_TRUE, CL_MAP_WRITE, 0, sobel_kern_size * sobel_kern_size, 0, NULL, NULL, &status);
    checkError(status, "Failed to map sobel_x kernel buffer to pointer");
    // kernel values copied from 3x3 from wikipedia: https://en.wikipedia.org/wiki/Kernel_(image_processing)
    *sobel_x_ptr = { 1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16 };
    clEnqueueUnmapMemObject(queue, sobel_x_cl, sobel_x_ptr, 0, NULL, NULL);

    // set sobel y convolution kernel
    float *sobel_y_ptr = (float *)clEnqueueMapBuffer(queue, sobel_y_cl, CL_TRUE, CL_MAP_WRITE, 0, sobel_kern_size * sobel_kern_size, 0, NULL, NULL, &status);
    checkError(status, "Failed to map sobel_x kernel buffer to pointer");
    // kernel values copied from 3x3 from wikipedia: https://en.wikipedia.org/wiki/Kernel_(image_processing)
    *sobel_y_ptr = { 1.0/16, 2.0/16, 1.0/16, 2.0/16, 4.0/16, 2.0/16, 1.0/16, 2.0/16, 1.0/16 };
    clEnqueueUnmapMemObject(queue, sobel_y_cl, sobel_y_ptr, 0, NULL, NULL);


    int max_frames = 100; // 299;
    while (true) {
        if (++count > max_frames) break;

        Mat cameraFrame;
        camera >> cameraFrame;
        cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);

        // do video filter on CPU using OpenCV
        auto start = chrono::high_resolution_clock::now();

#if GPU_GAUSSIAN
#else
        if (grayframe_ptr == NULL)
            grayframe_ptr = (unsigned char *)clEnqueueMapBuffer(queue, grayframe_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, frame_size_bytes, 0, NULL, NULL, &status);

        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
#endif  // GPU_GAUSSIAN


#if GPU_SOBEL
#else
        // remap these buffers to use on cpu
        if (edge_x_ptr == NULL)
            edge_x_ptr = (unsigned char *)clEnqueueMapBuffer(queue, edge_x_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, frame_size_bytes, 0, NULL, NULL, &status);

        if (edge_y_ptr == NULL)
            edge_y_ptr = (unsigned char *)clEnqueueMapBuffer(queue, edge_y_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, frame_size_bytes, 0, NULL, NULL, &status);

        Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT);
        Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT);
#endif  // GPU_SOBEL


        auto avg_start = chrono::high_resolution_clock::now();
#if GPU_AVERAGE
        clEnqueueUnmapMemObject(queue, edge_x_cl, edge_x_ptr, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, edge_y_cl, edge_y_ptr, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, edge_cl, edge_ptr, 0, NULL, NULL);
        edge_x_ptr = NULL; edge_y_ptr = NULL; edge_ptr = NULL;

        cl_event avg_event;
        status = clEnqueueNDRangeKernel(queue, average_kernel, 1, NULL, &frame_size_px, NULL, 0, NULL, &avg_event);
        checkError(status, "Failed to launch average kernel");

        status = clWaitForEvents(1, &avg_event);
        checkError(status, "Failed to wait for average event");

#else
        addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge);  // average between edge_x and edge_y, stored in edge
#endif  // GPU_AVERAGE
        auto avg_end = chrono::high_resolution_clock::now();
        auto avg_dur = chrono::duration_cast<chrono::microseconds>(avg_end - avg_start).count() / 1000.0f;


        auto thresh_start = chrono::high_resolution_clock::now();
#if GPU_THRESHOLD
        // launch threshold kernel
        clEnqueueUnmapMemObject(queue, edge_cl, edge_ptr, 0, NULL, NULL);
        edge_ptr = NULL;

        cl_event threshold_event;
        status = clEnqueueNDRangeKernel(queue, threshold_kernel, 1, NULL, &frame_size_px, NULL, 0, NULL, &threshold_event);
        checkError(status, "Failed to launch threshold kernel");

        status = clWaitForEvents(1, &threshold_event);
        checkError(status, "Failed to wait for threshold event");
#else
        if (edge_ptr == NULL)
            edge_ptr = (unsigned char *)clEnqueueMapBuffer(queue, edge_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, frame_size_bytes, 0, NULL, NULL, &status);

        threshold(edge, edge, THRESH_VAL, THRESH_MAXVAL, THRESH_BINARY_INV);  // threshold over 80, all data either 0 or 255
#endif  // GPU_THRESHOLD
        auto thresh_end = chrono::high_resolution_clock::now();
        auto thresh_dur = chrono::duration_cast<chrono::microseconds>(thresh_end - thresh_start).count() / 1000.0f;

        auto end = chrono::high_resolution_clock::now();


        if (edge_ptr == NULL)
            edge_ptr = (unsigned char *)clEnqueueMapBuffer(queue, edge_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, frame_size_bytes, 0, NULL, NULL, &status);

        Mat displayframe(size, CV_8U, grayframe_ptr);
        bitwise_and(displayframe, edge, displayframe);  // this also does masking
        outputVideo << displayframe;

#if SHOW
        imshow(window_name, displayframe);
        waitKey(1);
#endif
        
        auto diff = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0f;
        printf("avg: %.3f ms\tthresh: %.3f ms\tfull: %.3f ms\n", avg_dur, thresh_dur, diff);

        tot_ms += diff;
    }

    outputVideo.release();
    camera.release();
    printf("FPS %.2lf .\n", (1000.0f * max_frames)/tot_ms);

    return EXIT_SUCCESS;
}

