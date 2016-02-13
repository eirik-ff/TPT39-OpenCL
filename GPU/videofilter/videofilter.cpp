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

#define SHOW
#define STRING_BUFFER_LEN 1024


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

    // set static threshold args
    status = clSetKernelArg(threshold_kernel, 1, sizeof(int), &THRESH_VAL);
    checkError(status, "Failed to set thresh param in threshold kernel");
    status = clSetKernelArg(threshold_kernel, 2, sizeof(int), &THRESH_MAXVAL);
    checkError(status, "Failed to set maxval param in threshold kernel");


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

    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    int tot_ms = 0;
    int count = 0;
    const char *window_name = "filter";   // Name shown in the GUI window.

#ifdef SHOW
    namedWindow(window_name); // Resizable window, might not work on Windows.
    waitKey(1);
#endif

    size_t frame_size_px = size.width * size.height;
    size_t frame_size_bytes = frame_size_px * sizeof(unsigned char);


    cl_mem grayframe_cl, edge_x_cl, edge_y_cl, edge_cl;

    grayframe_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, frame_size_bytes, NULL, &status);
    checkError(status, "Failed to allocate grayframe buffer");

    edge_x_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, frame_size_bytes, NULL, &status);
    checkError(status, "Failed to allocate edge_x buffer");

    edge_y_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, frame_size_bytes, NULL, &status);
    checkError(status, "Failed to allocate edge_y buffer");

    edge_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, frame_size_bytes, NULL, &status);
    checkError(status, "Failed to allocate edge buffer");


    status = clSetKernelArg(threshold_kernel, 0, sizeof(cl_mem), &edge_cl);
    checkError(status, "Failed to set img param in threshold kernel");



    unsigned char *grayframe_ptr, *edge_x_ptr, *edge_y_ptr, *edge_ptr;

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


    int max_frames = 299;
    while (true) {
        if (++count > max_frames) break;

        Mat cameraFrame;
        camera >> cameraFrame;
        cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);

        // do video filter on CPU using OpenCV
        start = chrono::high_resolution_clock::now();

        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);

        Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT);
        Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT);

        addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge);  // average between edge_x and edge_y, stored in edge


#define CPU_THRESHOLD
        auto thresh_start = chrono::high_resolution_clock::now();
#ifdef CPU_THRESHOLD
        threshold(edge, edge, THRESH_VAL, THRESH_MAXVAL, THRESH_BINARY_INV);  // threshold over 80, all data either 0 or 255
#else
        // launch threshold kernel
        clEnqueueUnmapMemObject(queue, edge_cl, edge_ptr, 0, NULL, NULL);

        cl_event threshold_event;
        status = clEnqueueNDRangeKernel(queue, threshold_kernel, 1, NULL, &frame_size_px, NULL, 0, NULL, &threshold_event);
        checkError(status, "Failed to launch threshold kernel");

        status = clWaitForEvents(1, &threshold_event);
        checkError(status, "Failed to wait for threshold event");

        edge_ptr = (unsigned char *)clEnqueueMapBuffer(queue, edge_cl, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, frame_size_bytes, 0, NULL, NULL, &status);
#endif  // CPU_THRESHOLD
        auto thresh_end = chrono::high_resolution_clock::now();
        cout << "thresholding: " << chrono::duration_cast<chrono::microseconds>(thresh_end - thresh_start).count() / 1000.0f << " ms" << endl;

        end = chrono::high_resolution_clock::now();

        Mat displayframe(size, CV_8U, grayframe_ptr);
        bitwise_and(displayframe, edge, displayframe);  // this also does masking
        outputVideo << displayframe;

#ifdef SHOW
        imshow(window_name, displayframe);
        waitKey(1);
#endif
        
        diff = chrono::duration_cast<chrono::milliseconds>(end - start);
        tot_ms += diff.count();
    }

    outputVideo.release();
    camera.release();
    printf("FPS %.2lf .\n", (1000.0f * max_frames)/tot_ms);

    return EXIT_SUCCESS;
}

