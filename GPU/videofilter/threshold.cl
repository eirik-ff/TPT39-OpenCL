__kernel void threshold(__global unsigned char *img,
                        const int thresh,
                        const int maxval)
{
    int id = get_global_id(0);
    // this only implements THRESH_BINARY_INV from OpenCV
    img[id] = img[id] > thresh ? 0 : maxval;
}
