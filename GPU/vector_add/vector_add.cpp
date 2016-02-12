#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <chrono>

#define STRING_BUFFER_LEN 1024
#define USE_MAP_BUFFER 0

using namespace std;


void print_clbuild_errors(cl_program program,cl_device_id device)
{
    cout << "Program Build failed\n";
    size_t length;
    char buffer[2048];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            sizeof(buffer), buffer, &length);
    cout << "--- Build log ---\n "<< buffer << endl;
    exit(1);
}

unsigned char **read_file(const char *name) {
    size_t size;
    unsigned char **output=(unsigned char **)malloc(sizeof(unsigned char *));
    FILE* fp = fopen(name, "rb");
    if (!fp) {
        printf("no such file:%s",name);
        exit(-1);
    }

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *output = (unsigned char *)malloc(size);
    unsigned char **outputstr=(unsigned char **)malloc(sizeof(unsigned char *));
    *outputstr= (unsigned char *)malloc(size);
    if (!*output) {
        fclose(fp);
        printf("mem allocate failure:%s",name);
        exit(-1);
    }

    if(!fread(*output, size, 1, fp)) printf("failed to read file\n");
    fclose(fp);
    printf("file size %d\n",size);
    printf("-------------------------------------------\n");
    snprintf((char *)*outputstr,size,"%s\n",*output);
    printf("%s\n",*outputstr);
    printf("-------------------------------------------\n");
    return outputstr;
}
void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
    fwrite(buffer, 1, length, stdout);
}


void checkError(int status, const char *msg) {
    if(status!=CL_SUCCESS)	
        printf("%s\n",msg);
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
    return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

int main()
{
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
    cl_kernel kernel;

    //--------------------------------------------------------------------
    const unsigned long N = 50000000;
    int status;

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

    // build kernel
    unsigned char **opencl_program = read_file("vector_add.cl");
    program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
    if (program == NULL)
    {
        printf("Program creation failed\n");
        return 1;
    }	
    int success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(success != CL_SUCCESS) print_clbuild_errors(program,device);
    kernel = clCreateKernel(program, "vector_add", NULL);


    float *input_a = (float *)malloc(sizeof(float)*N);
    float *input_b = (float *)malloc(sizeof(float)*N);
    float *output = (float *)malloc(sizeof(float)*N);
    float *ref_output = (float *)malloc(sizeof(float)*N);
    cl_mem input_a_buf; // num_devices elements
    cl_mem input_b_buf; // num_devices elements
    cl_mem output_buf; // num_devices elements

    // create buffers
    auto start = chrono::high_resolution_clock::now();
    for(unsigned long j = 0; j < N; ++j) {
        input_a[j] = rand_float();
        input_b[j] = rand_float();
        //printf("ref %f\n",ref_output[j]);
    }
    auto end = chrono::high_resolution_clock::now();
    auto diff = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "It took " << diff.count() / 1000.0 << " ms to fill the buffers with random values." << endl;

    // time referene output on CPU
    start = chrono::high_resolution_clock::now();
    for(unsigned long j = 0; j < N; j++) {
        ref_output[j] = input_a[j] + input_b[j];
    }
    end = chrono::high_resolution_clock::now();
    diff = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "CPU took " << diff.count() / 1000.0 << " ms to run." << endl;

    // run on GPU

#if USE_MAP_BUFFER
    // Input buffers.
    input_a_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
            N * sizeof(float), (void *)input_a, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
            N * sizeof(float), (void *)input_b, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR,
            N * sizeof(float), (void *)output, &status);
    checkError(status, "Failed to create buffer for output");
#else
    // Input buffers.
    input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
            N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
            N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");
#endif // USE_MAP_BUFFER


#if USE_MAP_BUFFER
    // memory map buffers instead of copying them
    cl_event write_event[2];

    void *input_a_buf_ptr = clEnqueueMapBuffer(queue, input_a_buf, CL_TRUE, 
        CL_MAP_READ, 0, N * sizeof(float), 0, NULL, &write_event[0], &status);
    checkError(status, "Failed to map input buffer a.");

    void *input_b_buf_ptr = clEnqueueMapBuffer(queue, input_b_buf, CL_TRUE, 
        CL_MAP_READ, 0, N * sizeof(float), 0, NULL, &write_event[1], &status);
    checkError(status, "Failed to map input buffer b.");

    void *output_buf_ptr = clEnqueueMapBuffer(queue, output_buf, CL_TRUE,
        CL_MAP_WRITE, 0, N * sizeof(float), 0, NULL, NULL, &status);
    checkError(status, "Failed to map output buffer.");
#else
    start = chrono::high_resolution_clock::now();

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
    status = clEnqueueWriteBuffer(queue, input_a_buf, CL_TRUE,
            0, N* sizeof(float), input_a, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue, input_b_buf, CL_TRUE,
            0, N* sizeof(float), input_b, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");

    end = chrono::high_resolution_clock::now();
    diff = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Copying buffers from CPU to GPU took " << diff.count() / 1000.0f << " ms." << endl;
#endif /* USE_MAP_BUFFER */

    // Set kernel arguments.
    cl_event kernel_event, finish_event;
    unsigned argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3");

    const size_t global_work_size = N;
    start = chrono::high_resolution_clock::now();

    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
            &global_work_size, NULL, 2, write_event, &kernel_event);
    checkError(status, "Failed to launch kernel");

    status = clWaitForEvents(1, &kernel_event);
    checkError(status, "Failed wait");

    end = chrono::high_resolution_clock::now();
    diff = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "GPU took " << diff.count() / 1000.0 << " ms to run. (no read buffer)" << endl;


    start = chrono::high_resolution_clock::now();
    cl_event kernel_event2;
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
            &global_work_size, NULL, 2, write_event, &kernel_event2);
    checkError(status, "Failed to launch kernel");

    status = clWaitForEvents(1, &kernel_event2);
    checkError(status, "Failed wait");

    end = chrono::high_resolution_clock::now();
    diff = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "GPU2 took " << diff.count() / 1000.0 << " ms to run. (no read buffer)" << endl;

    // Read the result. This the final operation.
    start = chrono::high_resolution_clock::now();

    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
            0, N* sizeof(float), output, 1, &kernel_event, &finish_event);

    end = chrono::high_resolution_clock::now();
    diff = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "It took " << diff.count() / 1000.0 << " ms to read buffer." << endl;

    // Verify results.
    bool pass = true;
    for(unsigned long j = 0; j < N && pass; ++j) {
        if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
            printf("Failed verification @ index %ld\nOutput: %f\nReference: %f\n",
                    j, output[j], ref_output[j]);
            pass = false;
        }
    }

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(input_a_buf);
    clReleaseMemObject(input_b_buf);
    clReleaseMemObject(output_buf);
    clReleaseProgram(program);
    clReleaseContext(context);


    //--------------------------------------------------------------------


    clFinish(queue);

    return 0;
}

