__kernel void vector_add(__global const float4 *x, 
                         __global const float4 *y, 
                         __global float4 *restrict z)
{
    int id = get_global_id(0);
    z[id] = x[id] + y[id];
}

