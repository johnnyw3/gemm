#include <iostream>
#include <stdint.h>
#include <string.h>
#include "simd_common.h"

void amx_relayout(__bf16 *b_mat, int n_col, int n_row)
{
    __bf16 *tmp = (__bf16*)malloc(sizeof(__bf16) * n_col * n_row);

    for (int outer_y = 0; outer_y < n_row; outer_y += 32)
    {
        for (int outer_x = 0; outer_x < n_col; outer_x += 16)
        {
            for (int inner_y = 0; inner_y < 32; ++inner_y)
            {
                for (int inner_x = 0; inner_x < 16; ++inner_x)
                {
                    tmp[(outer_y + inner_y)/2*n_col*2 + (outer_x + inner_x)*2 + inner_y % 2] =
                        b_mat[(outer_y + inner_y)*n_col + (outer_x + inner_x)];
                }
            }
        }
    }

    memcpy(b_mat, tmp, sizeof(__bf16) * n_col * n_row);
}
