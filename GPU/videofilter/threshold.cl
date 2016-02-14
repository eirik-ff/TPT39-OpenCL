__kernel void threshold(__global uchar16 *img,
                        const int thresh,
                        const int maxval)
{
    // this only implements THRESH_BINARY_INV from OpenCV
    int idx = get_global_id(0);
    uchar16 th = (uchar16)(thresh);
    uchar16 mv = (uchar16)(maxval);
    img[idx] = img[idx] > th ? (uchar16)(0) : mv;
    // img[idx] = img[idx] > thresh ? 0 : maxval;
}
