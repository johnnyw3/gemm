#include <iostream>
#include <stdint.h>
#include <string.h>
#include "simd_common.h"

__bf16 *amx_relayout(__bf16 *b_mat, int n_col, int n_row)
{
    __bf16 *tmp = (__bf16*)malloc(sizeof(__bf16) * n_col * n_row);

    for (int outer_y = 0; outer_y < n_row; outer_y += 2)
    {
        for (int outer_x = 0; outer_x < n_col; outer_x += 32)
        {
            __bf16 *top_baseaddr = b_mat + outer_y*n_col + outer_x;
            __bf16 *bot_baseaddr = b_mat + (outer_y + 1)*n_col + outer_x;
            __m512i topl = _mm512_maskz_expandloadu_epi16(0x55555555, top_baseaddr);
            __m512i topr = _mm512_maskz_expandloadu_epi16(0x55555555, top_baseaddr + 16);
            __m512i botl = _mm512_maskz_expandloadu_epi16(0xAAAAAAAA, bot_baseaddr);
            __m512i botr = _mm512_maskz_expandloadu_epi16(0xAAAAAAAA, bot_baseaddr + 16);

            __m512h resl = _mm512_mask_blend_ph(0xAAAAAAAA, (__m512h)topl, (__m512h)botl);
            __m512h resr = _mm512_mask_blend_ph(0xAAAAAAAA, (__m512h)topr, (__m512h)botr);

            __bf16 *out_baseaddr = tmp + outer_y*n_col + outer_x*2;
            _mm512_storeu_ph(out_baseaddr, resl);
            _mm512_storeu_ph(out_baseaddr + 32, resr);
        }
    }

    return tmp;
}
