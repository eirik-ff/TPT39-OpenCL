#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <math.h>
#include <time.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <chrono>

#include "helpers.h"

#define STRING_BUFFER_LEN 1024

#define USE_2D_KERNEL 1


using namespace std;

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
    return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

void set_matrix_elem(float *mat, float elem, int row, int col, int N) {
    int idx = row * N + col;
    mat[idx] = elem;
}

float get_matrix_elem(float *mat, int row, int col, int N) {
    int idx = row * N + col;
    return mat[idx];
}

void print_matrix(float *mat, int N, const char *name) {
    if (mat == NULL) {
        printf("[print_matrix] Input matrix 'mat' is NULL\n");
        return;
    }

    printf("%s = \n", name);
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            printf("%.4f, ", get_matrix_elem(mat, row, col, N));
        }
        printf("\n");
    }
}


// calculates the matrix product C = A*B
void matrix_mul_cpu(float *C, float *A, float *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idxC = i * N + j;  // C[i, j]
            C[idxC] = 0;

            for (int k = 0; k < N; k++) {
                int idxA = i * N + k;  // A[i, k]
                int idxB = k * N + j;  // B[k, j]
                C[idxC] += A[idxA] * B[idxB];
            }
        }
    }
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

    // Define dimensions of two matrices A: NxN and B: NxN
    const long N = 1000;
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
#if USE_2D_KERNEL
#define KERNEL_DIM 2
    const char *program_file_name = "matrix_mul_2d.cl";
    const size_t global_work_size[2] = {N, N};
#else
#define KERNEL_DIM 1
    const char *program_file_name = "matrix_mul_1d.cl";
    const size_t ws = N*N;
    const size_t *global_work_size = &ws;
#endif // USE_2D_KERNEL
    unsigned char **opencl_program = read_file(program_file_name);
    program = clCreateProgramWithSource(context, 1, (const char **)opencl_program, NULL, NULL);
    if (program == NULL)
    {
        printf("Program creation failed\n");
        return 1;
    }	
    int success = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(success != CL_SUCCESS) print_clbuild_errors(program,device);
    kernel = clCreateKernel(program, "matrix_mul", NULL);
    printf("Kernel build successful\n");


    // matrices are stored as a contiguous block of memory in row-major order,
    // i.e. mat = [row1, row2, ..., rowN]
    // to index element (row, col) use equation idx = row * N + col;
    // for 1 <= K <= N, rowK is slice K * N : (K+1) * N
    cl_mem matA_cl; // num_devices elements
    cl_mem matB_cl; // num_devices elements
    cl_mem output_cl; // num_devices elements

    float *matA = NULL;
    float *matB = NULL;
    float *output = NULL;
    float *ref_output = (float *)malloc(N * N * sizeof(float));

    cl_event write_event[2];

    // malloc on gpu
    // see developer guide file:///cal/exterieurs/ath-8669/Downloads/arm_mali_midgard_opencl_developer_guide_100614_0313_00_en.pdf
    // for why se use CL_MEM_ALLOC_HOST_PTR 
    matA_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, N * N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create and alloc input A buffer.");

    matB_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, N * N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create and alloc input B buffer.");

    output_cl = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, N * N * sizeof(float), NULL, &status);
    checkError(status, "Failed to create and alloc output buffer.");


    // map input arguments for writing
    matA = (float *)clEnqueueMapBuffer(queue, matA_cl, CL_TRUE, CL_MAP_WRITE, 0, N * N * sizeof(float), 0, NULL, &write_event[0], &status);
    checkError(status, "Failed to map input buffer A.");

    matB = (float *)clEnqueueMapBuffer(queue, matB_cl, CL_TRUE, CL_MAP_WRITE, 0, N * N * sizeof(float), 0, NULL, &write_event[1], &status);
    checkError(status, "Failed to map input buffer B.");



    // fill buffer with random values
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < N * N; i++) {
        matA[i] = rand_float();
        matB[i] = rand_float();
    }
    auto end = chrono::high_resolution_clock::now();
    auto diff = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "It took " << diff.count() / 1000.0f << " ms to fill the buffers with random values." << endl;

    // time referene output on CPU
    start = chrono::high_resolution_clock::now();
    matrix_mul_cpu(ref_output, matA, matB, N);
    end = chrono::high_resolution_clock::now();
    diff = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "CPU took " << diff.count() / 1000.0f << " ms to run." << endl;

    // we need to unmap the memory regions before launching the kernel 
    // see https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/clEnqueueUnmapMemObject.html
    // for more information
    clEnqueueUnmapMemObject(queue, matA_cl, matA, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, matB_cl, matB, 0, NULL, NULL);


    // Set kernel arguments.
    cl_event kernel_event;
    unsigned int argi = 0;

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &matA_cl);
    checkError(status, "Failed to set argument 1");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &matB_cl);
    checkError(status, "Failed to set argument 2");

    status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_cl);
    checkError(status, "Failed to set argument 3");

    status = clSetKernelArg(kernel, argi++, sizeof(int), &N);
    checkError(status, "Failed to set argument 4");



    // first run
    start = chrono::high_resolution_clock::now();

    status = clEnqueueNDRangeKernel(queue, kernel, KERNEL_DIM, NULL, global_work_size, NULL, 0, NULL, &kernel_event);
    checkError(status, "Failed to launch kernel");

    status = clWaitForEvents(1, &kernel_event);
    checkError(status, "Failed wait");

    end = chrono::high_resolution_clock::now();
    diff = chrono::duration_cast<chrono::microseconds>(end - start);
#if USE_2D_KERNEL
        cout << "GPU took " << diff.count() / 1000.0 << " ms to run (no read buffer) using 2D kernel" << endl;
#else
        cout << "GPU took " << diff.count() / 1000.0 << " ms to run (no read buffer) using 1D kernel" << endl;
#endif  // USE_2D_KERNEL

    // second run
//    start = chrono::high_resolution_clock::now();
//    cl_event kernel_event2;
//    status = clEnqueueNDRangeKernel(queue, kernel, KERNEL_DIM, NULL, global_work_size, NULL, 0, NULL, &kernel_event);
//    checkError(status, "Failed to launch kernel 2");
//
//    status = clWaitForEvents(1, &kernel_event2);
//    checkError(status, "Failed wait 2");
//
//    end = chrono::high_resolution_clock::now();
//    diff = chrono::duration_cast<chrono::microseconds>(end - start);
// #if USE_2D_KERNEL
//         cout << "GPU took " << diff.count() / 1000.0 << " ms to run (no read buffer) using 2D kernel" << endl;
// #else
//         cout << "GPU took " << diff.count() / 1000.0 << " ms to run (no read buffer) using 1D kernel" << endl;
// #endif  // USE_2D_KERNEL


    // only need to map the output buffer when we want to read the input
    output = (float *)clEnqueueMapBuffer(queue, output_cl, CL_TRUE, CL_MAP_READ, 0, N * N * sizeof(float), 0, NULL, NULL, &status);
    checkError(status, "Failed to map output buffer.");

    // Verify results.
    bool pass = true;
    for(int j = 0; j < N * N && pass; ++j) {
        if(fabsf(output[j] - ref_output[j]) > 1.0e-5f) {
            printf("Failed verification @ index %d\nOutput: %f\nReference: %f\n", j, output[j], ref_output[j]);
            pass = false;
        }
    }

    if (pass)
        printf("Output and reference are equal\n");

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(matA_cl);
    clReleaseMemObject(matB_cl);
    clReleaseMemObject(output_cl);
    clReleaseProgram(program);
    clReleaseContext(context);
    free(ref_output);


    clFinish(queue);

    return 0;
}

