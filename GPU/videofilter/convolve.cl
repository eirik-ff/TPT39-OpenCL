#define KERN_SIZE 3
#define HALF_SIZE 1
#define WIDTH 640
#define HEIGHT 360

__kernel void convolve(__global const unsigned char *in,
                       __global unsigned char *out,
                       const int width, 
                       __constant float *kern) 
{
    // assuming kernel is square
    int gid = get_global_id(0);
    int row = gid / WIDTH;
    int col = gid % WIDTH;

    // kern_size must be odd, so half_size is even
    float res = 0;
    for (int i = -HALF_SIZE; i <= HALF_SIZE; i++) {
        for (int j = -HALF_SIZE; j <= HALF_SIZE; j++) {
            // res += img[row + i, col + j] * kern[i, j];

            int local_row = row + i;
            int local_col = col + j;
            int img_idx = local_row * WIDTH + local_col;

            unsigned char px;
            px = in[img_idx];

            int kern_idx = (HALF_SIZE + i) * KERN_SIZE + (HALF_SIZE + j);

            res += px * kern[kern_idx];
        }
    }

    out[gid] = (unsigned char)res;
}

