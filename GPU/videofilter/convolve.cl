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
    for (int i = -HALF_SIZE; i <= HALF_SIZE; i++) {
        for (int j = -HALF_SIZE; j <= HALF_SIZE; j++) {
            // res += img[row + i, col + j] * kern[i, j];

            int local_row = row + i;
            int local_col = col + j;
            int img_idx = local_row * width + local_col;

            unsigned char px;
            px = in[img_idx];

            int kern_row = (HALF_SIZE + i);
            int kern_col = (HALF_SIZE + j);
            int kern_idx = kern_row * KERN_SIZE + kern_col;

            res += px * kern[kern_idx];
        }
    }

    out[gid] = (unsigned char)res;
}

