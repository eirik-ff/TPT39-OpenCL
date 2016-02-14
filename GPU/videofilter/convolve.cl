kernel void convolve(global float *img,
                     const int width, 
                     const int height,
                     global const float *kern,
                     const int kern_size) 
{
    // assuming kernel is square
    int gid = get_global_id(0);
    int row = gid / width;
    int col = gid % width;
    
    // kern_size must be odd, so half_kern_size is even
    int half_kern_size = kern_size / 2;  
    float res = 0;
    for (int i = -half_kern_size; i < half_kern_size; i++) {
        for (int j = -half_kern_size; j < half_kern_size; j++) {
            // res += img[row + i, col + j] * kern[i, j];
            int local_row = row + i;
            int local_col = col + j;
            int img_idx = local_row * width + local_col;

            unsigned char px;
            if (local_row < 0 || local_row >= width || local_col < 0 || local_col >= height)
                px = 0;
            else
                px = img[img_idx];

            int kern_idx = (half_kern_size + i) * kern_size + (half_kern_size + j);

            res += px * kern[kern_idx];
        }
    }

    img[gid] = (unsigned char)res;
}
                       
