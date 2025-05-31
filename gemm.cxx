#include <iostream>
#include <stdint.h>
#include "simd_common.h"

void cpu_transpose(float *mat, int n)
{
    for (int idx_y = 0; idx_y < n; ++idx_y)
    {
        for (int idx_x = idx_y+1; idx_x < n; ++idx_x)
        {
            float temp_upper = *(mat + idx_y*n + idx_x);
            *(mat + idx_y*n + idx_x) = *(mat + idx_x*n + idx_y);
            *(mat + idx_x*n + idx_y) = temp_upper;
        }
    }
}
