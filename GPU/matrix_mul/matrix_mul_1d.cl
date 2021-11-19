// 1D kernel
__kernel void matrix_mul(__global const float *A, 
                         __global const float *B, 
                         __global float *restrict C,
                         const int N)
{
    int gid = get_global_id(0);
    int i = gid / N;
    int j = gid % N;

    int idxC = i * N + j;
    C[idxC] = 0;

    for (int k = 0; k < N; k++) {
        int idxA = i * N + k;
        int idxB = k * N + j;

        C[idxC] += A[idxA] * B[idxB];
    }
}

