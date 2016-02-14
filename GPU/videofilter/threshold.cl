kernel void threshold(global unsigned char *img,
                      const int thresh,
                      const int maxval)
{
    // this only implements THRESH_BINARY_INV from OpenCV
    int idx = get_global_id(0);
    img[idx] = img[idx] > thresh ? 0 : maxval;
}
