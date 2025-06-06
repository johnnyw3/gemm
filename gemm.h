#ifndef __GEMM_H__
#define __GEMM_H__ 1
#include <x86intrin.h>
#include <pthread.h>
#include <string.h>

#define US_PER_S 1000000
#define GIGA     1000000000

#define BLOCK_WIDTH 256 // in bytes -> 128x128 block
                        // a 64x64 block of floats uses 16K of memory (64KB L1d cache on this CPU - i5-8350u)

template<typename T>
void simd_gemm(T *mat1, T *mat2, T *dst, int n);
void* simd_gemm_worker(void *argv);
void* simd_gemm_worker2(void *argv);

inline void gemm_inner(float *mat1_ptr, float *mat2_ptr, float *dst_ptr, int simd_ele_width, int block_ele_width);
inline void gemm_inner(double *mat1_ptr, double *mat2_ptr, double *dst_ptr, int simd_ele_width, int block_ele_width);
void cpu_transpose(float *mat, int n);

typedef struct{
    float *mat1;
    float *mat2;
    float *dst;
    int    n;
    int    th_id;
} gemmOptns;

void simd_gemm(float * __restrict mat1, float * __restrict mat2, float * __restrict dst, int n)
{
    gemmOptns thd_optns[NUM_THREADS];
    pthread_t thds[NUM_THREADS];

    for (int th_id = 0; th_id < NUM_THREADS; ++th_id)
    {
        gemmOptns optn =  {mat1, mat2, dst, n, th_id};
        thd_optns[th_id] = optn;
        pthread_create(&thds[th_id], NULL, &simd_gemm_worker, (void*)&thd_optns[th_id]);

    }

    for (int th_id = 0; th_id < NUM_THREADS; ++th_id)
    {
        pthread_join(thds[th_id], NULL);
    }

}

void* simd_gemm_worker(void *argv)
{
    gemmOptns *optns = (gemmOptns*)argv;
    float *mat1 = optns->mat1;
    float *mat2 = optns->mat2;
    float *dst  = optns->dst;
    int    n    = optns->n;
    int    th_id = optns->th_id;
    int    thd_loop_sz = n / NUM_THREADS;
    int    start_idx = thd_loop_sz * th_id;
    int    stop_idx  = start_idx + thd_loop_sz;

    int simd_ele_width  = SIMD_WIDTH  / sizeof(float);
    int block_ele_width = BLOCK_WIDTH / sizeof(float);
    int vec_n = n / simd_ele_width;

    float * __restrict mat1_ptr, * __restrict mat2_ptr, * __restrict dst_ptr;
    float * __restrict mat2_ptr2, * __restrict mat2_ptr3, * __restrict mat2_ptr4;

    for (int i_outer = start_idx; i_outer < stop_idx; i_outer += block_ele_width)
    {
        for (int j_outer = 0; j_outer < n; j_outer += block_ele_width)
        {
            for (int k_outer = 0; k_outer < n; k_outer += block_ele_width)
            {
                float packed_b[block_ele_width*block_ele_width] __attribute__ ((__aligned__(64)));
                float packed_a[block_ele_width*block_ele_width] __attribute__ ((__aligned__(64)));

                for (int idx = 0; idx < block_ele_width; ++idx)
                {
                    memcpy(packed_b + idx*block_ele_width, mat2 + (j_outer + idx)*n + k_outer, BLOCK_WIDTH);
                    memcpy(packed_a + idx*block_ele_width, mat1 + (i_outer + idx)*n + k_outer, BLOCK_WIDTH);
                }

                for (int i_inner = 0; i_inner < block_ele_width; ++i_inner)
                {
                    mat1_ptr = packed_a + (i_inner)*block_ele_width;
                    //_mm_prefetch(mat1_ptr, _MM_HINT_T0);

                    //dst_ptr = dst_tmp + i_inner*block_ele_width; // + (i_outer + i_inner)*n + j_outer;
                    dst_ptr = dst + (i_outer + i_inner)*n + j_outer;
                    //_mm_prefetch(dst_ptr, _MM_HINT_T0); 
                    
                    for (int j_inner = 0; j_inner < block_ele_width; j_inner += simd_ele_width)
                    {

                            __m256 a_vec, b_vec, b_vec2, b_vec3, b_vec4;
                            __m256 dst2 = _mm256_setzero_ps();

                            __m256 sums[simd_ele_width];
                            for (int idx = 0; idx < simd_ele_width; ++idx)
                                sums[idx] = _mm256_setzero_ps();

                            mat2_ptr  = packed_b + (j_inner + 0)*block_ele_width;
                            mat2_ptr2 = packed_b + (j_inner + 1)*block_ele_width;
                            mat2_ptr3 = packed_b + (j_inner + 2)*block_ele_width;
                            mat2_ptr4 = packed_b + (j_inner + 3)*block_ele_width;
                            for (int k_inner = 0; k_inner < block_ele_width; k_inner += simd_ele_width)
                            {
                                a_vec = _mm256_load_ps( mat1_ptr + k_inner );
                                b_vec = _mm256_load_ps( mat2_ptr + k_inner );
                                sums[0] = _mm256_fmadd_ps(a_vec, b_vec, sums[0]);
                                b_vec2 = _mm256_load_ps( mat2_ptr2 + k_inner );
                                sums[1] = _mm256_fmadd_ps(a_vec, b_vec2, sums[1]);
                                b_vec3 = _mm256_load_ps( mat2_ptr3 + k_inner );
                                sums[2] = _mm256_fmadd_ps(a_vec, b_vec3, sums[2]);
                                b_vec4 = _mm256_load_ps( mat2_ptr4 + k_inner );
                                sums[3] = _mm256_fmadd_ps(a_vec, b_vec4, sums[3]);
                            }

                            __m256 lower, upper, hsum, shuf;
                            mat2_ptr  = packed_b + (j_inner + 4)*block_ele_width;
                            mat2_ptr2 = packed_b + (j_inner + 5)*block_ele_width;
                            mat2_ptr3 = packed_b + (j_inner + 6)*block_ele_width;
                            mat2_ptr4 = packed_b + (j_inner + 7)*block_ele_width;
                                a_vec = _mm256_load_ps( mat1_ptr + 0 );
                                b_vec = _mm256_load_ps( mat2_ptr + 0 );
                                sums[4] = _mm256_fmadd_ps(a_vec, b_vec, sums[4]);
                                b_vec2 = _mm256_load_ps( mat2_ptr2 + 0 );
                                sums[5] = _mm256_fmadd_ps(a_vec, b_vec2, sums[5]);
                                b_vec3 = _mm256_load_ps( mat2_ptr3 + 0 );
                                sums[6] = _mm256_fmadd_ps(a_vec, b_vec3, sums[6]);
                                b_vec4 = _mm256_load_ps( mat2_ptr4 + 0 );
                                sums[7] = _mm256_fmadd_ps(a_vec, b_vec4, sums[7]);
                            lower = _mm256_permute2f128_ps(sums[0], sums[1], 0x20);
                            upper = _mm256_permute2f128_ps(sums[0], sums[1], 0x31);
                            hsum  = _mm256_add_ps(lower, upper);
                            shuf  = _mm256_permute_ps(hsum, 0x1B);
                                   hsum  = _mm256_add_ps(hsum, shuf);
                                   shuf  = _mm256_permute_ps(hsum, 0xB1);
                                   hsum  = _mm256_add_ps(hsum, shuf);
                            dst2 = _mm256_blend_ps(dst2, hsum, 0x21);
                            
                                a_vec = _mm256_load_ps( mat1_ptr + simd_ele_width );
                                b_vec = _mm256_load_ps( mat2_ptr + simd_ele_width );
                                sums[4] = _mm256_fmadd_ps(a_vec, b_vec, sums[4]);
                                b_vec2 = _mm256_load_ps( mat2_ptr2 + simd_ele_width );
                                sums[5] = _mm256_fmadd_ps(a_vec, b_vec2, sums[5]);
                                b_vec3 = _mm256_load_ps( mat2_ptr3 + simd_ele_width );
                                sums[6] = _mm256_fmadd_ps(a_vec, b_vec3, sums[6]);
                                b_vec4 = _mm256_load_ps( mat2_ptr4 + simd_ele_width );
                                sums[7] = _mm256_fmadd_ps(a_vec, b_vec4, sums[7]);
                            lower = _mm256_permute2f128_ps(sums[2], sums[3], 0x20);
                            upper = _mm256_permute2f128_ps(sums[2], sums[3], 0x31);
                            hsum  = _mm256_add_ps(lower, upper);
                            shuf  = _mm256_permute_ps(hsum, 0x1B);
                            hsum  = _mm256_add_ps(hsum, shuf);
                            shuf  = _mm256_permute_ps(hsum, 0xB1);
                            hsum  = _mm256_add_ps(hsum, shuf);
                            dst2  = _mm256_blend_ps(dst2, hsum, 0x84);
                            for (int k_inner = simd_ele_width*2; k_inner < block_ele_width; k_inner += simd_ele_width)
                            {
                                a_vec = _mm256_load_ps( mat1_ptr + k_inner );
                                b_vec = _mm256_load_ps( mat2_ptr + k_inner );
                                sums[4] = _mm256_fmadd_ps(a_vec, b_vec, sums[4]);
                                b_vec2 = _mm256_load_ps( mat2_ptr2 + k_inner );
                                sums[5] = _mm256_fmadd_ps(a_vec, b_vec2, sums[5]);
                                b_vec3 = _mm256_load_ps( mat2_ptr3 + k_inner );
                                sums[6] = _mm256_fmadd_ps(a_vec, b_vec3, sums[6]);
                                b_vec4 = _mm256_load_ps( mat2_ptr4 + k_inner );
                                sums[7] = _mm256_fmadd_ps(a_vec, b_vec4, sums[7]);
                            }


                            lower = _mm256_permute2f128_ps(sums[4], sums[5], 0x20);
                            upper = _mm256_permute2f128_ps(sums[4], sums[5], 0x31);
                            hsum  = _mm256_add_ps(lower, upper);
                            lower = _mm256_permute2f128_ps(sums[6], sums[7], 0x20);
                            upper = _mm256_permute2f128_ps(sums[6], sums[7], 0x31);
                            shuf  = _mm256_permute_ps(hsum, 0x1B);
                            hsum  = _mm256_add_ps(hsum, shuf);
                            shuf  = _mm256_permute_ps(hsum, 0xB1);
                            hsum  = _mm256_add_ps(hsum, shuf);
                            dst2  = _mm256_blend_ps(dst2, hsum, 0x12);

                            hsum  = _mm256_add_ps(lower, upper);
                            shuf  = _mm256_permute_ps(hsum, 0x1B);
                            hsum  = _mm256_add_ps(hsum, shuf);
                            shuf  = _mm256_permute_ps(hsum, 0xB1);
                            hsum  = _mm256_add_ps(hsum, shuf);
                            dst2  = _mm256_blend_ps(dst2, hsum, 0x48);

                            __m256 swapd = _mm256_permute2f128_ps(dst2, dst2, 0x21);
                            dst2 = _mm256_blend_ps(dst2, swapd, 0xAA);
                            swapd = _mm256_permute_ps(dst2, 0xB1);
                            dst2 = _mm256_blend_ps(dst2, swapd, 0xF0);

                        //__m256 sums = _mm256_load_ps(temp); 
                        //__m256 sums = _mm256_setzero_ps(); 
                        __m256 dst_vec = _mm256_load_ps(dst_ptr);
                        dst2 = _mm256_add_ps(dst2, dst_vec);
                        _mm256_store_ps(dst_ptr, dst2);
                        dst_ptr += simd_ele_width;
                        
                    }
                }
            }
        }
    }
    return NULL;
}

#endif  // __GEMM_H__
