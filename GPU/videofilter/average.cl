__kernel void average(__global const uchar4 *in1, 
                      __global const uchar4 *in2,
                      __global uchar4 *out)
{
    int gid = get_global_id(0);
    uchar4 div = (uchar4)(2);
    out[gid] = in1[gid] / div + in2[gid] / div;
}
