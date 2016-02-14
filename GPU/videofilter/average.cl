kernel void average(global unsigned char *in1, 
                    global unsigned char *in2,
                    global unsigned char *out)
{
    int gid = get_global_id(0);
    out[gid] = in1[gid] / 2 + in2[gid] / 2;
}
