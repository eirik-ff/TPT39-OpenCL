#define KERN_SIZE 3
#define HALF_SIZE 1

// this kernel only works for 3x3 kernels because there was
// a massive speedup by hardcoding the kernel sizes
__kernel void convolve(__global const unsigned char *in,
                       __global unsigned char *out,
                       const int width, 
                       __constant float *kern) 
{
    int gid = get_global_id(0);
    int row = gid / width;
    int col = gid % width;

    float res = 0;

    // there doesn't seem to be any speedup from doing manual loop unrolling
    res += in[(row - 1) * width + (col - 1)] * kern[(HALF_SIZE - 1) * KERN_SIZE + (HALF_SIZE - 1)];
    res += in[(row - 1) * width + (col + 0)] * kern[(HALF_SIZE - 1) * KERN_SIZE + (HALF_SIZE + 0)];
    res += in[(row - 1) * width + (col + 1)] * kern[(HALF_SIZE - 1) * KERN_SIZE + (HALF_SIZE + 1)];

    res += in[(row + 0) * width + (col - 1)] * kern[(HALF_SIZE + 0) * KERN_SIZE + (HALF_SIZE - 1)];
    res += in[(row + 0) * width + (col + 0)] * kern[(HALF_SIZE + 0) * KERN_SIZE + (HALF_SIZE + 0)];
    res += in[(row + 0) * width + (col + 1)] * kern[(HALF_SIZE + 0) * KERN_SIZE + (HALF_SIZE + 1)];

    res += in[(row + 1) * width + (col - 1)] * kern[(HALF_SIZE + 1) * KERN_SIZE + (HALF_SIZE - 1)];
    res += in[(row + 1) * width + (col + 0)] * kern[(HALF_SIZE + 1) * KERN_SIZE + (HALF_SIZE + 0)];
    res += in[(row + 1) * width + (col + 1)] * kern[(HALF_SIZE + 1) * KERN_SIZE + (HALF_SIZE + 1)];

    out[gid] = (unsigned char)res;
}

