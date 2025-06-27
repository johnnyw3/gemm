#ifndef __GEMM_H__
#define __GEMM_H__ 1
#include <x86intrin.h>
#include <pthread.h>
#include <string.h>

#define US_PER_S 1000000
#define GIGA     1000000000

template<typename T>
void simd_gemm(T *mat1, T *mat2, T *dst, int n);
void* simd_gemm_worker(void *argv);
void* simd_gemm_worker_avx512(void *argv);

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
#ifdef __AVX512F__
        pthread_create(&thds[th_id], NULL, &simd_gemm_worker_avx512, (void*)&thd_optns[th_id]);
#else 
        pthread_create(&thds[th_id], NULL, &simd_gemm_worker, (void*)&thd_optns[th_id]);
#endif 

    }

    for (int th_id = 0; th_id < NUM_THREADS; ++th_id)
    {
        pthread_join(thds[th_id], NULL);
    }

}

void* simd_gemm_worker(void *argv)
{
    gemmOptns *optns = (gemmOptns*)argv;
    float * const mat1 = optns->mat1;
    float * const mat2 = optns->mat2;
    float * const dst  = optns->dst;
    const int    n    = optns->n;
    const int    th_id = optns->th_id;
    const int    thd_loop_sz = n / NUM_THREADS;
    const int    start_idx = thd_loop_sz * th_id;
    const int    stop_idx  = start_idx + thd_loop_sz;

    constexpr int simd_ele_width  = SIMD_WIDTH  / sizeof(float);
    constexpr int block_ele_i = BLOCK_I / sizeof(float);
    constexpr int block_ele_j = BLOCK_J / sizeof(float);
    constexpr int block_ele_k = BLOCK_K / sizeof(float);
    constexpr int sblock_ele_i = SBLOCK_I / sizeof(float);
    constexpr int sblock_ele_j = SBLOCK_J / sizeof(float);
    constexpr int sblock_ele_k = SBLOCK_K / sizeof(float);
    //int vec_n = n / simd_ele_width;
    constexpr int block_ni = sblock_ele_i/block_ele_i;
    constexpr int block_nj = sblock_ele_j/block_ele_j;
    constexpr int block_nk = sblock_ele_k/block_ele_k;

    float * __restrict mat1_ptr, * __restrict mat2_ptr, * __restrict dst_ptr;
    float * __restrict mat1_ptr2, * __restrict dst_ptr2;
    float * __restrict mat2_ptr2, * __restrict mat2_ptr3, * __restrict mat2_ptr4;

    for (int i_outer = start_idx; i_outer < stop_idx; i_outer += sblock_ele_i)
    {
        for (int j_outer = 0; j_outer < n; j_outer += sblock_ele_j)
        {
            for (int k_outer = 0; k_outer < n; k_outer += sblock_ele_k)
            {
                float packed_b[sblock_ele_j*sblock_ele_k] __attribute__ ((__aligned__(64)));
                float packed_a[sblock_ele_k*sblock_ele_i] __attribute__ ((__aligned__(64)));

                mat1_ptr = mat1 + (i_outer)*n + k_outer;
                for (int idx = 0; idx < sblock_ele_i; ++idx)
                {
                    mat2_ptr = packed_a + (idx%block_ele_i) * block_ele_k + (idx/block_ele_i)*block_ele_i*block_ele_k*block_nk;
                    for (int jdx = 0; jdx < sblock_ele_k; jdx += block_ele_k)
                    {
                    //memcpy(packed_b + jdx*block_ele_width + (idx%block_ele_width) * block_ele_width + (idx/block_ele_width)*block_ele_width*block_ele_width*block_n, mat2 + (j_outer + idx)*n + jdx + k_outer, BLOCK_WIDTH);
                    memcpy(mat2_ptr, mat1_ptr + jdx, BLOCK_K);
                    mat2_ptr += block_ele_k*block_ele_i;
                    }
                    mat1_ptr += n;
                }

                mat2_ptr = mat2 + (j_outer)*n + k_outer;
                for (int idx = 0; idx < sblock_ele_j; ++idx)
                {
                    mat1_ptr = packed_b + (idx%block_ele_j) * block_ele_k + (idx/block_ele_j)*block_ele_k*block_ele_j*block_nk;
                    for (int jdx = 0; jdx < sblock_ele_k; jdx += block_ele_k)
                    {
                    memcpy(mat1_ptr, mat2_ptr + jdx, BLOCK_K);
                    //memcpy(packed_a + jdx*block_ele_width + (idx%block_ele_width) * block_ele_width + (idx/block_ele_width)*block_ele_width*block_ele_width*block_n, mat1 + (i_outer + idx)*n + jdx + k_outer, BLOCK_WIDTH);
                    mat1_ptr += block_ele_k*block_ele_j;
                    }
                    mat2_ptr += n;
                }

    for (int i_outer2 = 0; i_outer2< sblock_ele_i; i_outer2+= block_ele_i)
    {
        for (int j_outer2= 0; j_outer2< sblock_ele_j; j_outer2 += block_ele_j)
        {
#if 0        
            float packed_dst[block_ele_i*block_ele_j];
            for (int idx = 0; idx < block_ele_i; ++idx)
            {
                dst_ptr = dst + (i_outer + i_outer2+ idx)*n + j_outer + j_outer2;
                memcpy(packed_dst + idx*block_ele_j, dst_ptr, BLOCK_J);
            }
#endif            
            for (int k_outer2= 0; k_outer2< sblock_ele_k; k_outer2 += block_ele_k)
            {
                for (int i_inner = 0; i_inner < block_ele_i; i_inner += 2)
                {
                    mat1_ptr = packed_a + (i_outer2*block_nk + i_inner)*block_ele_k + k_outer2*block_ele_i;
                    mat1_ptr2 = mat1_ptr + block_ele_k;
                    //_mm_prefetch(mat1_ptr, _MM_HINT_T0);

                    //dst_ptr = dst_tmp + i_inner*block_ele_width; // + (i_outer + i_inner)*n + j_outer;
                    dst_ptr = dst + (i_outer + i_outer2 + i_inner)*n + j_outer + j_outer2;
                    //dst_ptr = packed_dst + i_inner*block_ele_j;
                    dst_ptr2 = dst_ptr + n;
                    //dst_ptr2 = dst_ptr + block_ele_j;
                    //_mm_prefetch(dst_ptr, _MM_HINT_T0); 
                    
                    for (int j_inner = 0; j_inner < block_ele_j; j_inner += simd_ele_width)
                    {

                            __m256 a_vec, a_vec2, b_vec, b_vec2, b_vec3, b_vec4;
                            __m256 dst2, dst3; // = _mm256_setzero_ps();

                            //__m256 sums8 = {};
                            //__m256 sums28 = {};
                            //for (int idx = 0; idx < simd_ele_width; ++idx)
                            __m256 sums0 = _mm256_setzero_ps();
                            __m256 sums1 = _mm256_setzero_ps();
                            __m256 sums2 = _mm256_setzero_ps();
                            __m256 sums3 = _mm256_setzero_ps();
                            __m256 sums4 = _mm256_setzero_ps();
                            __m256 sums5 = _mm256_setzero_ps();
                            __m256 sums6 = _mm256_setzero_ps();
                            __m256 sums7 = _mm256_setzero_ps();
                            __m256 sums20 = _mm256_setzero_ps();
                            __m256 sums21 = _mm256_setzero_ps();
                            __m256 sums22 = _mm256_setzero_ps();
                            __m256 sums23 = _mm256_setzero_ps();
                            __m256 sums24 = _mm256_setzero_ps();
                            __m256 sums25 = _mm256_setzero_ps();
                            __m256 sums26 = _mm256_setzero_ps();
                            __m256 sums27 = _mm256_setzero_ps();
                            //    sums[idx] = _mm256_setzero_ps();

                            mat2_ptr  = packed_b + (j_inner + j_outer2*block_nk + 0)*block_ele_k + k_outer2*block_ele_j;
                            mat2_ptr2 = mat2_ptr + block_ele_k;
                            mat2_ptr3 = mat2_ptr2 + block_ele_k;
                            mat2_ptr4 = mat2_ptr3 + block_ele_k;
                            //mat2_ptr3 = packed_b + (j_inner + j_outer2*block_nk + 2)*block_ele_k + k_outer2*block_ele_j;
                            //mat2_ptr4 = packed_b + (j_inner + j_outer2*block_nk + 3)*block_ele_k + k_outer2*block_ele_j;
                            for (int k_inner = 0; k_inner < block_ele_k; k_inner += simd_ele_width)
                            {
                                a_vec = _mm256_load_ps( mat1_ptr + k_inner );
                                a_vec2 = _mm256_load_ps( mat1_ptr2 + k_inner );

                                b_vec = _mm256_load_ps( mat2_ptr + k_inner );
                                b_vec2 = _mm256_load_ps( mat2_ptr2 + k_inner );
                                b_vec3 = _mm256_load_ps( mat2_ptr3+ k_inner );
                                b_vec4 = _mm256_load_ps( mat2_ptr4 + k_inner );
                                sums0 = _mm256_fmadd_ps(a_vec, b_vec, sums0);
                                sums20 = _mm256_fmadd_ps(a_vec2, b_vec, sums20);

                                sums1 = _mm256_fmadd_ps(a_vec, b_vec2, sums1);
                                sums21 = _mm256_fmadd_ps(a_vec2, b_vec2, sums21);

                                sums2 = _mm256_fmadd_ps(a_vec, b_vec3, sums2);
                                sums22 = _mm256_fmadd_ps(a_vec2, b_vec3, sums22);

                                sums3 = _mm256_fmadd_ps(a_vec, b_vec4, sums3);
                                sums23 = _mm256_fmadd_ps(a_vec2, b_vec4, sums23);
                            }

                            __m256 const upper = _mm256_unpacklo_ps(sums0, sums2); // 0100 1110
                            __m256 const lower = _mm256_unpackhi_ps(sums0, sums2);
                            __m256 const upper2 = _mm256_unpacklo_ps(sums1, sums3); // 0100 1110
                            __m256 const lower2 = _mm256_unpackhi_ps(sums1, sums3);
                            __m256 const upper1 = _mm256_unpacklo_ps(sums20, sums22); // 0100 1110
                            __m256 const lower1 = _mm256_unpackhi_ps(sums20, sums22);
                            __m256 const upper12 = _mm256_unpacklo_ps(sums21, sums23); // 0100 1110
                            __m256 const lower12 = _mm256_unpackhi_ps(sums21, sums23);
                            __m256 const res4 = _mm256_add_ps(lower, upper);
                            __m256 const res5 = _mm256_add_ps(lower2, upper2);
                            __m256 const res14= _mm256_add_ps(lower1, upper1);
                            __m256 const res15= _mm256_add_ps(lower12, upper12);

                            __m256 const upper3 = _mm256_unpacklo_ps(res4, res5); //_mm256_shuffle_ps(sums0, sums2, 0x4E);
                            __m256 const lower3 = _mm256_unpackhi_ps(res4, res5); //_mm256_blend_ps(sums0, sums2, 0xCC);
                            __m256 const upper13 = _mm256_unpacklo_ps(res14, res15); //_mm256_shuffle_ps(sums20, sums22, 0x4E ); // 0100 1110
                            __m256 const lower13 = _mm256_unpackhi_ps(res14, res15); //_mm256_blend_ps(sums20, sums22, 0xCC ); // 1100 1100
                            dst2 = _mm256_add_ps(lower3, upper3);
                            dst3 = _mm256_add_ps(lower13, upper13);


                            mat2_ptr  = packed_b + (j_inner + j_outer2*block_nk + 4)*block_ele_k + k_outer2*block_ele_j;
                            mat2_ptr2 = mat2_ptr + block_ele_k;
                            mat2_ptr3 = mat2_ptr2 + block_ele_k;
                            mat2_ptr4 = mat2_ptr3 + block_ele_k;
                            for (int k_inner = 0; k_inner < block_ele_k; k_inner += simd_ele_width)
                            {
                                a_vec = _mm256_load_ps( mat1_ptr + k_inner );
                                a_vec2 = _mm256_load_ps( mat1_ptr2 + k_inner );

                                b_vec = _mm256_load_ps( mat2_ptr + k_inner );
                                b_vec2 = _mm256_load_ps( mat2_ptr2 + k_inner );
                                b_vec3 = _mm256_load_ps( mat2_ptr3 + k_inner );
                                b_vec4 = _mm256_load_ps( mat2_ptr4 + k_inner );

                                sums4 = _mm256_fmadd_ps(a_vec, b_vec, sums4);
                                sums24 = _mm256_fmadd_ps(a_vec2, b_vec, sums24);

                                sums5 = _mm256_fmadd_ps(a_vec, b_vec2, sums5);
                                sums25 = _mm256_fmadd_ps(a_vec2, b_vec2, sums25);

                                sums6 = _mm256_fmadd_ps(a_vec, b_vec3, sums6);
                                sums26 = _mm256_fmadd_ps(a_vec2, b_vec3, sums26);

                                sums7 = _mm256_fmadd_ps(a_vec, b_vec4, sums7);
                                sums27 = _mm256_fmadd_ps(a_vec2, b_vec4, sums27);
                            }

                            __m256 const upper4 = _mm256_unpacklo_ps(sums4, sums6); // 0100 1110
                            __m256 const lower4 = _mm256_unpackhi_ps(sums4, sums6);
                            __m256 const upper5 = _mm256_unpacklo_ps(sums5, sums7); // 0100 1110
                            __m256 const lower5 = _mm256_unpackhi_ps(sums5, sums7);
                            __m256 const upper14 = _mm256_unpacklo_ps(sums24, sums26); // 0100 1110
                            __m256 const lower14 = _mm256_unpackhi_ps(sums24, sums26);
                            __m256 const upper15 = _mm256_unpacklo_ps(sums25, sums27); // 0100 1110
                            __m256 const lower15 = _mm256_unpackhi_ps(sums25, sums27);
                            __m256 const res1 = _mm256_add_ps(lower4, upper4);
                            __m256 const res2 = _mm256_add_ps(lower5, upper5);
                            __m256 const res11 = _mm256_add_ps(lower14, upper14);
                            __m256 const res12 = _mm256_add_ps(lower15, upper15);


                        __m256 const upper18 = _mm256_load_ps(dst_ptr2);
                            __m256 const upper6 =  _mm256_unpacklo_ps(res1, res2); //_mm256_shuffle_ps(sums4, sums6, 0x4E);
                            __m256 const lower6 = _mm256_unpackhi_ps(res1, res2); //_mm256_blend_ps(sums4, sums6, 0xCC);
                            __m256 const upper16 = _mm256_unpacklo_ps(res11, res12); //_mm256_shuffle_ps(sums24, sums26, 0x4E);
                            __m256 const lower16 = _mm256_unpackhi_ps(res11, res12); //_mm256_blend_ps(sums24, sums26, 0xCC);
                            __m256 const res3 = _mm256_add_ps(lower6, upper6);
                            __m256 const res13= _mm256_add_ps(lower16, upper16);


                            __m256 const lower17 = _mm256_permute2f128_ps(dst3, res13, 0x21 ); // 1101 1101 => 0xDD
                        __m256 const upper8 = _mm256_load_ps(dst_ptr);
                            __m256 const lower7 = _mm256_permute2f128_ps(dst2, res3 , 0x21); // 1101 1101 => 0xDD
                            __m256 const upper17 = _mm256_blend_ps(dst3, res13, 0xF0); // 1000 1000 => 0x88
                            __m256 const upper7 = _mm256_blend_ps(dst2, res3, 0xF0); // 1000 1000 => 0x88
                            dst2 = _mm256_add_ps(lower7, upper7);
                            dst3 = _mm256_add_ps(lower17, upper17);

                        //__m256 sums2 = _mm256_load_ps(temp); 
                        //__m256 sums2 = _mm256_setzero_ps(); 
                        dst2 = _mm256_add_ps(dst2, upper8);
                        dst3 = _mm256_add_ps(dst3, upper18);
                        _mm256_store_ps(dst_ptr, dst2);
                        _mm256_store_ps(dst_ptr2, dst3);
                        dst_ptr += simd_ele_width;
                        dst_ptr2 += simd_ele_width;

                       }}} 
#if 0                       
                for (int idx = 0; idx < block_ele_i; ++idx)
                {
                    dst_ptr = dst + (i_outer + i_outer2 + idx)*n + j_outer + j_outer2;
                    memcpy(dst_ptr, packed_dst + idx*block_ele_j, BLOCK_J);
                }
#endif                
                    }
                }
            }
        }
    }
    return NULL;
}

void* simd_gemm_worker_avx512(void *argv)
{
    gemmOptns *optns = (gemmOptns*)argv;
    float * const mat1 = optns->mat1;
    float * const mat2 = optns->mat2;
    float * const dst  = optns->dst;
    const int    n    = optns->n;
    const int    th_id = optns->th_id;
    const int    thd_loop_sz = n / NUM_THREADS;
    const int    start_idx = thd_loop_sz * th_id;
    const int    stop_idx  = start_idx + thd_loop_sz;

    constexpr int simd_ele_width  = SIMD_WIDTH  / sizeof(float);
    constexpr int block_ele_i = BLOCK_I / sizeof(float);
    constexpr int block_ele_j = BLOCK_J / sizeof(float);
    constexpr int block_ele_k = BLOCK_K / sizeof(float);
    constexpr int sblock_ele_i = SBLOCK_I / sizeof(float);
    constexpr int sblock_ele_j = SBLOCK_J / sizeof(float);
    constexpr int sblock_ele_k = SBLOCK_K / sizeof(float);
    //int vec_n = n / simd_ele_width;
    constexpr int block_ni = sblock_ele_i/block_ele_i;
    constexpr int block_nj = sblock_ele_j/block_ele_j;
    constexpr int block_nk = sblock_ele_k/block_ele_k;

    float * __restrict mat1_ptr, * __restrict mat2_ptr, * __restrict dst_ptr;
    float * __restrict mat1_ptr2, * __restrict dst_ptr2;
    float * __restrict mat2_ptr2, * __restrict mat2_ptr3, * __restrict mat2_ptr4;

    for (int i_outer = start_idx; i_outer < stop_idx; i_outer += sblock_ele_i)
    {
        for (int j_outer = 0; j_outer < n; j_outer += sblock_ele_j)
        {
            for (int k_outer = 0; k_outer < n; k_outer += sblock_ele_k)
            {
                float packed_b[sblock_ele_j*sblock_ele_k] __attribute__ ((__aligned__(64)));
                float packed_a[sblock_ele_k*sblock_ele_i] __attribute__ ((__aligned__(64)));

                mat1_ptr = mat1 + (i_outer)*n + k_outer;
                for (int idx = 0; idx < sblock_ele_i; ++idx)
                {
                    mat2_ptr = packed_a + (idx%block_ele_i) * block_ele_k + (idx/block_ele_i)*block_ele_i*block_ele_k*block_nk;
                    for (int jdx = 0; jdx < sblock_ele_k; jdx += block_ele_k)
                    {
                    //memcpy(packed_b + jdx*block_ele_width + (idx%block_ele_width) * block_ele_width + (idx/block_ele_width)*block_ele_width*block_ele_width*block_n, mat2 + (j_outer + idx)*n + jdx + k_outer, BLOCK_WIDTH);
                    memcpy(mat2_ptr, mat1_ptr + jdx, BLOCK_K);
                    mat2_ptr += block_ele_k*block_ele_i;
                    }
                    mat1_ptr += n;
                }

                mat2_ptr = mat2 + (j_outer)*n + k_outer;
                for (int idx = 0; idx < sblock_ele_j; ++idx)
                {
                    mat1_ptr = packed_b + (idx%block_ele_j) * block_ele_k + (idx/block_ele_j)*block_ele_k*block_ele_j*block_nk;
                    for (int jdx = 0; jdx < sblock_ele_k; jdx += block_ele_k)
                    {
                    memcpy(mat1_ptr, mat2_ptr + jdx, BLOCK_K);
                    //memcpy(packed_a + jdx*block_ele_width + (idx%block_ele_width) * block_ele_width + (idx/block_ele_width)*block_ele_width*block_ele_width*block_n, mat1 + (i_outer + idx)*n + jdx + k_outer, BLOCK_WIDTH);
                    mat1_ptr += block_ele_k*block_ele_j;
                    }
                    mat2_ptr += n;
                }

    for (int i_outer2 = 0; i_outer2< sblock_ele_i; i_outer2+= block_ele_i)
    {
        for (int j_outer2= 0; j_outer2< sblock_ele_j; j_outer2 += block_ele_j)
        {
#if 0        
            float packed_dst[block_ele_i*block_ele_j];
            for (int idx = 0; idx < block_ele_i; ++idx)
            {
                dst_ptr = dst + (i_outer + i_outer2+ idx)*n + j_outer + j_outer2;
                memcpy(packed_dst + idx*block_ele_j, dst_ptr, BLOCK_J);
            }
#endif            
            for (int k_outer2= 0; k_outer2< sblock_ele_k; k_outer2 += block_ele_k)
            {
                for (int i_inner = 0; i_inner < block_ele_i; i_inner += 2)
                {
                    mat1_ptr = packed_a + (i_outer2*block_nk + i_inner)*block_ele_k + k_outer2*block_ele_i;
                    mat1_ptr2 = mat1_ptr + block_ele_k;
                    //_mm_prefetch(mat1_ptr, _MM_HINT_T0);

                    //dst_ptr = dst_tmp + i_inner*block_ele_width; // + (i_outer + i_inner)*n + j_outer;
                    dst_ptr = dst + (i_outer + i_outer2 + i_inner)*n + j_outer + j_outer2;
                    //dst_ptr = packed_dst + i_inner*block_ele_j;
                    dst_ptr2 = dst_ptr + n;
                    //dst_ptr2 = dst_ptr + block_ele_j;
                    //_mm_prefetch(dst_ptr, _MM_HINT_T0); 
                    
                    for (int j_inner = 0; j_inner < block_ele_j; j_inner += simd_ele_width)
                    {

                            __m512 a_vec, a_vec2, b_vec, b_vec2, b_vec3, b_vec4;
                            __m512 dst2, dst3; // = _mm512_setzero_ps();

                            //__m512 sums8 = {};
                            //__m512 sums28 = {};
                            //for (int idx = 0; idx < simd_ele_width; ++idx)
                            __m512 sums0 = _mm512_setzero_ps();
                            __m512 sums1 = _mm512_setzero_ps();
                            __m512 sums2 = _mm512_setzero_ps();
                            __m512 sums3 = _mm512_setzero_ps();
                            __m512 sums4 = _mm512_setzero_ps();
                            __m512 sums5 = _mm512_setzero_ps();
                            __m512 sums6 = _mm512_setzero_ps();
                            __m512 sums7 = _mm512_setzero_ps();
                            __m512 sums8 = _mm512_setzero_ps();
                            __m512 sums9 = _mm512_setzero_ps();
                            __m512 sumsA = _mm512_setzero_ps();
                            __m512 sumsB = _mm512_setzero_ps();
                            __m512 sumsC = _mm512_setzero_ps();
                            __m512 sumsD = _mm512_setzero_ps();
                            __m512 sumsE = _mm512_setzero_ps();
                            __m512 sumsF = _mm512_setzero_ps();
                            __m512 sums20 = _mm512_setzero_ps();
                            __m512 sums21 = _mm512_setzero_ps();
                            __m512 sums22 = _mm512_setzero_ps();
                            __m512 sums23 = _mm512_setzero_ps();
                            __m512 sums24 = _mm512_setzero_ps();
                            __m512 sums25 = _mm512_setzero_ps();
                            __m512 sums26 = _mm512_setzero_ps();
                            __m512 sums27 = _mm512_setzero_ps();
                            __m512 sums28 = _mm512_setzero_ps();
                            __m512 sums29 = _mm512_setzero_ps();
                            __m512 sums2A = _mm512_setzero_ps();
                            __m512 sums2B = _mm512_setzero_ps();
                            __m512 sums2C = _mm512_setzero_ps();
                            __m512 sums2D = _mm512_setzero_ps();
                            __m512 sums2E = _mm512_setzero_ps();
                            __m512 sums2F = _mm512_setzero_ps();
                            //    sums[idx] = _mm512_setzero_ps();

                            mat2_ptr  = packed_b + (j_inner + j_outer2*block_nk + 0)*block_ele_k + k_outer2*block_ele_j;
                            mat2_ptr2 = mat2_ptr + block_ele_k;
                            mat2_ptr3 = mat2_ptr2 + block_ele_k;
                            mat2_ptr4 = mat2_ptr3 + block_ele_k;
                            //mat2_ptr3 = packed_b + (j_inner + j_outer2*block_nk + 2)*block_ele_k + k_outer2*block_ele_j;
                            //mat2_ptr4 = packed_b + (j_inner + j_outer2*block_nk + 3)*block_ele_k + k_outer2*block_ele_j;
                            for (int k_inner = 0; k_inner < block_ele_k; k_inner += simd_ele_width)
                            {
                                a_vec = _mm512_load_ps( mat1_ptr + k_inner );
                                a_vec2 = _mm512_load_ps( mat1_ptr2 + k_inner );

                                b_vec = _mm512_load_ps( mat2_ptr + k_inner );
                                b_vec2 = _mm512_load_ps( mat2_ptr2 + k_inner );
                                b_vec3 = _mm512_load_ps( mat2_ptr3+ k_inner );
                                b_vec4 = _mm512_load_ps( mat2_ptr4 + k_inner );
                                sums0 = _mm512_fmadd_ps(a_vec, b_vec, sums0);
                                sums20 = _mm512_fmadd_ps(a_vec2, b_vec, sums20);

                                sums1 = _mm512_fmadd_ps(a_vec, b_vec2, sums1);
                                sums21 = _mm512_fmadd_ps(a_vec2, b_vec2, sums21);

                                sums2 = _mm512_fmadd_ps(a_vec, b_vec3, sums2);
                                sums22 = _mm512_fmadd_ps(a_vec2, b_vec3, sums22);

                                sums3 = _mm512_fmadd_ps(a_vec, b_vec4, sums3);
                                sums23 = _mm512_fmadd_ps(a_vec2, b_vec4, sums23);
                            }

                            __m512 const upper = _mm512_unpacklo_ps(sums0, sums2); // 0100 1110
                            __m512 const lower = _mm512_unpackhi_ps(sums0, sums2);
                            __m512 const upper2 = _mm512_unpacklo_ps(sums1, sums3); // 0100 1110
                            __m512 const lower2 = _mm512_unpackhi_ps(sums1, sums3);
                            __m512 const upper1 = _mm512_unpacklo_ps(sums20, sums22); // 0100 1110
                            __m512 const lower1 = _mm512_unpackhi_ps(sums20, sums22);
                            __m512 const upper12 = _mm512_unpacklo_ps(sums21, sums23); // 0100 1110
                            __m512 const lower12 = _mm512_unpackhi_ps(sums21, sums23);
                            __m512 const res000 = _mm512_add_ps(lower, upper);
                            __m512 const res001 = _mm512_add_ps(lower2, upper2);
                            __m512 const res100 = _mm512_add_ps(lower1, upper1);
                            __m512 const res101 = _mm512_add_ps(lower12, upper12);

                            __m512 const upper3 = _mm512_unpacklo_ps(res000, res001); //_mm512_shuffle_ps(sums0, sums2, 0x4E);
                            __m512 const lower3 = _mm512_unpackhi_ps(res000, res001); //_mm512_blend_ps(sums0, sums2, 0xCC);
                            __m512 const upper13 = _mm512_unpacklo_ps(res100, res101); //_mm512_shuffle_ps(sums20, sums22, 0x4E ); // 0100 1110
                            __m512 const lower13 = _mm512_unpackhi_ps(res100, res101); //_mm512_blend_ps(sums20, sums22, 0xCC ); // 1100 1100
                            __m512 const res010 = _mm512_add_ps(lower3, upper3);
                            __m512 const res110 = _mm512_add_ps(lower13, upper13);


                            mat2_ptr  = packed_b + (j_inner + j_outer2*block_nk + 8)*block_ele_k + k_outer2*block_ele_j;
                            mat2_ptr2 = mat2_ptr + block_ele_k;
                            mat2_ptr3 = mat2_ptr2 + block_ele_k;
                            mat2_ptr4 = mat2_ptr3 + block_ele_k;
                            for (int k_inner = 0; k_inner < block_ele_k; k_inner += simd_ele_width)
                            {
                                a_vec = _mm512_load_ps( mat1_ptr + k_inner );
                                a_vec2 = _mm512_load_ps( mat1_ptr2 + k_inner );

                                b_vec = _mm512_load_ps( mat2_ptr + k_inner );
                                b_vec2 = _mm512_load_ps( mat2_ptr2 + k_inner );
                                b_vec3 = _mm512_load_ps( mat2_ptr3 + k_inner );
                                b_vec4 = _mm512_load_ps( mat2_ptr4 + k_inner );

                                sums8 = _mm512_fmadd_ps(a_vec, b_vec, sums8);
                                sums28 = _mm512_fmadd_ps(a_vec2, b_vec, sums28);

                                sums9 = _mm512_fmadd_ps(a_vec, b_vec2, sums9);
                                sums29 = _mm512_fmadd_ps(a_vec2, b_vec2, sums29);

                                sumsA = _mm512_fmadd_ps(a_vec, b_vec3, sumsA);
                                sums2A = _mm512_fmadd_ps(a_vec2, b_vec3, sums2A);

                                sumsB = _mm512_fmadd_ps(a_vec, b_vec4, sumsB);
                                sums2B = _mm512_fmadd_ps(a_vec2, b_vec4, sums2B);
                            }

                            __m512 const upper4 = _mm512_unpacklo_ps(sums8, sumsA); // 0100 1110
                            __m512 const lower4 = _mm512_unpackhi_ps(sums8, sumsA);
                            __m512 const upper5 = _mm512_unpacklo_ps(sums9, sumsB); // 0100 1110
                            __m512 const lower5 = _mm512_unpackhi_ps(sums9, sumsB);
                            __m512 const upper14 = _mm512_unpacklo_ps(sums28, sums2A); // 0100 1110
                            __m512 const lower14 = _mm512_unpackhi_ps(sums28, sums2A);
                            __m512 const upper15 = _mm512_unpacklo_ps(sums29, sums2B); // 0100 1110
                            __m512 const lower15 = _mm512_unpackhi_ps(sums29, sums2B);
                            __m512 const res002 = _mm512_add_ps(lower4, upper4);
                            __m512 const res003 = _mm512_add_ps(lower5, upper5);
                            __m512 const res102 = _mm512_add_ps(lower14, upper14);
                            __m512 const res103 = _mm512_add_ps(lower15, upper15);


                            __m512 const upper6 =  _mm512_unpacklo_ps(res002, res003); //_mm512_shuffle_ps(sums4, sums6, 0x4E);
                            __m512 const lower6 = _mm512_unpackhi_ps(res002, res003); //_mm512_blend_ps(sums4, sums6, 0xCC);
                            __m512 const upper16 = _mm512_unpacklo_ps(res102, res103); //_mm512_shuffle_ps(sums24, sums26, 0x4E);
                            __m512 const lower16 = _mm512_unpackhi_ps(res102, res103); //_mm512_blend_ps(sums24, sums26, 0xCC);
                            __m512 const res011 = _mm512_add_ps(lower6, upper6);
                            __m512 const res111 = _mm512_add_ps(lower16, upper16);

                            __m512i const permutemask0 = _mm512_set_epi32(0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10, 0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08);
                            __m512 const upper7 = _mm512_permutex2var_ps(res010, permutemask0, res011);
                            __m512 const upper17 = _mm512_permutex2var_ps(res110, permutemask0, res111);
                            __m512 const lower7 = _mm512_mask_blend_ps(0xFF00, res010, res011);
                            __m512 const lower17 = _mm512_mask_blend_ps(0xFF00, res110, res111);
                            __m512 const res020 = _mm512_add_ps(lower7, upper7);
                            __m512 const res120 = _mm512_add_ps(lower17, upper17);

                            mat2_ptr  = packed_b + (j_inner + j_outer2*block_nk + 4)*block_ele_k + k_outer2*block_ele_j;
                            mat2_ptr2 = mat2_ptr + block_ele_k;
                            mat2_ptr3 = mat2_ptr2 + block_ele_k;
                            mat2_ptr4 = mat2_ptr3 + block_ele_k;
                            for (int k_inner = 0; k_inner < block_ele_k; k_inner += simd_ele_width)
                            {
                                a_vec = _mm512_load_ps( mat1_ptr + k_inner );
                                a_vec2 = _mm512_load_ps( mat1_ptr2 + k_inner );

                                b_vec = _mm512_load_ps( mat2_ptr + k_inner );
                                b_vec2 = _mm512_load_ps( mat2_ptr2 + k_inner );
                                b_vec3 = _mm512_load_ps( mat2_ptr3 + k_inner );
                                b_vec4 = _mm512_load_ps( mat2_ptr4 + k_inner );

                                sums4 = _mm512_fmadd_ps(a_vec, b_vec, sums4);
                                sums24 = _mm512_fmadd_ps(a_vec2, b_vec, sums24);

                                sums5 = _mm512_fmadd_ps(a_vec, b_vec2, sums5);
                                sums25 = _mm512_fmadd_ps(a_vec2, b_vec2, sums25);

                                sums6 = _mm512_fmadd_ps(a_vec, b_vec3, sums6);
                                sums26 = _mm512_fmadd_ps(a_vec2, b_vec3, sums26);

                                sums7 = _mm512_fmadd_ps(a_vec, b_vec4, sums7);
                                sums27 = _mm512_fmadd_ps(a_vec2, b_vec4, sums27);
                            }
                            __m512 const upper8 = _mm512_unpacklo_ps(sums4, sums6); // 0100 1110
                            __m512 const lower8 = _mm512_unpackhi_ps(sums4, sums6);
                            __m512 const upper9 = _mm512_unpacklo_ps(sums5, sums7); // 0100 1110
                            __m512 const lower9 = _mm512_unpackhi_ps(sums5, sums7);
                            __m512 const upper18 = _mm512_unpacklo_ps(sums24, sums26); // 0100 1110
                            __m512 const lower18 = _mm512_unpackhi_ps(sums24, sums26);
                            __m512 const upper19 = _mm512_unpacklo_ps(sums25, sums27); // 0100 1110
                            __m512 const lower19 = _mm512_unpackhi_ps(sums25, sums27);
                            __m512 const res004 = _mm512_add_ps(lower8, upper8);
                            __m512 const res005 = _mm512_add_ps(lower9, upper9);
                            __m512 const res104 = _mm512_add_ps(lower18, upper18);
                            __m512 const res105 = _mm512_add_ps(lower19, upper19);

                            __m512 const upperA =  _mm512_unpacklo_ps(res004, res005); //_mm512_shuffle_ps(sums4, sums6, 0x4E);
                            __m512 const lowerA = _mm512_unpackhi_ps(res004, res005); //_mm512_blend_ps(sums4, sums6, 0xCC);
                            __m512 const upper1A = _mm512_unpacklo_ps(res104, res105); //_mm512_shuffle_ps(sums24, sums26, 0x4E);
                            __m512 const lower1A = _mm512_unpackhi_ps(res104, res105); //_mm512_blend_ps(sums24, sums26, 0xCC);
                            __m512 const res012 = _mm512_add_ps(lowerA, upperA);
                            __m512 const res112 = _mm512_add_ps(lower1A, upper1A);


                            mat2_ptr  = packed_b + (j_inner + j_outer2*block_nk + 12)*block_ele_k + k_outer2*block_ele_j;
                            mat2_ptr2 = mat2_ptr + block_ele_k;
                            mat2_ptr3 = mat2_ptr2 + block_ele_k;
                            mat2_ptr4 = mat2_ptr3 + block_ele_k;
                            for (int k_inner = 0; k_inner < block_ele_k; k_inner += simd_ele_width)
                            {
                                a_vec = _mm512_load_ps( mat1_ptr + k_inner );
                                a_vec2 = _mm512_load_ps( mat1_ptr2 + k_inner );

                                b_vec = _mm512_load_ps( mat2_ptr + k_inner );
                                b_vec2 = _mm512_load_ps( mat2_ptr2 + k_inner );
                                b_vec3 = _mm512_load_ps( mat2_ptr3 + k_inner );
                                b_vec4 = _mm512_load_ps( mat2_ptr4 + k_inner );

                                sumsC = _mm512_fmadd_ps(a_vec, b_vec, sumsC);
                                sums2C = _mm512_fmadd_ps(a_vec2, b_vec, sums2C);

                                sumsD = _mm512_fmadd_ps(a_vec, b_vec2, sumsD);
                                sums2D = _mm512_fmadd_ps(a_vec2, b_vec2, sums2D);

                                sumsE = _mm512_fmadd_ps(a_vec, b_vec3, sumsE);
                                sums2E = _mm512_fmadd_ps(a_vec2, b_vec3, sums2E);

                                sumsF = _mm512_fmadd_ps(a_vec, b_vec4, sumsF);
                                sums2F = _mm512_fmadd_ps(a_vec2, b_vec4, sums2F);
                            }
                            __m512 const upperC = _mm512_unpacklo_ps(sumsC, sumsE); // 0100 1110
                            __m512 const lowerC = _mm512_unpackhi_ps(sumsC, sumsE);
                            __m512 const upperD = _mm512_unpacklo_ps(sumsD, sumsF); // 0100 1110
                            __m512 const lowerD = _mm512_unpackhi_ps(sumsD, sumsF);
                            __m512 const upper1C = _mm512_unpacklo_ps(sums2C, sums2E); // 0100 1110
                            __m512 const lower1C = _mm512_unpackhi_ps(sums2C, sums2E);
                            __m512 const upper1D = _mm512_unpacklo_ps(sums2D, sums2F); // 0100 1110
                            __m512 const lower1D = _mm512_unpackhi_ps(sums2D, sums2F);
                            __m512 const res006 = _mm512_add_ps(lowerC, upperC);
                            __m512 const res007 = _mm512_add_ps(lowerD, upperD);
                            __m512 const res106 = _mm512_add_ps(lower1C, upper1C);
                            __m512 const res107 = _mm512_add_ps(lower1D, upper1D);

                            __m512 const upperE =  _mm512_unpacklo_ps(res006, res007); //_mm512_shuffle_ps(sums4, sums6, 0x4E);
                            __m512 const lowerE = _mm512_unpackhi_ps(res006, res007); //_mm512_blend_ps(sums4, sums6, 0xCC);
                            __m512 const upper1E = _mm512_unpacklo_ps(res106, res107); //_mm512_shuffle_ps(sums24, sums26, 0x4E);
                            __m512 const lower1E = _mm512_unpackhi_ps(res106, res107); //_mm512_blend_ps(sums24, sums26, 0xCC);
                            __m512 const res013 = _mm512_add_ps(lowerE, upperE);
                            __m512 const res113 = _mm512_add_ps(lower1E, upper1E);

                            __m512 const upper11118 = _mm512_load_ps(dst_ptr2);
                            __m512 const upper118 = _mm512_load_ps(dst_ptr);

                            __m512 const upperF = _mm512_permutex2var_ps(res012, permutemask0, res013);
                            __m512 const upper1F = _mm512_permutex2var_ps(res112, permutemask0, res113);
                            __m512 const lowerF = _mm512_mask_blend_ps(0xFF00, res012, res013);
                            __m512 const lower1F = _mm512_mask_blend_ps(0xFF00, res112, res113);
                            __m512 const res021 = _mm512_add_ps(lowerF, upperF);
                            __m512 const res121 = _mm512_add_ps(lower1F, upper1F);

                            __m512i permutemask2 = _mm512_set_epi32(0x1B, 0x1A, 0x19, 0x18, 0x0F, 0x0E, 0x0D, 0x0C, 0x13, 0x12, 0x11, 0x10, 0x07, 0x06, 0x05, 0x04);
                            __m512 const lower0 = _mm512_permutex2var_ps(res020, permutemask2, res021);
                            __m512 const lower10 = _mm512_permutex2var_ps(res120, permutemask2, res121);
                            __m512 const upper0 = _mm512_mask_blend_ps(0xF0F0, res020, res021);
                            __m512 const upper10 = _mm512_mask_blend_ps(0xF0F0, res120, res121);
                            dst2 = _mm512_add_ps(lower0, upper0);
                            dst3 = _mm512_add_ps(lower10, upper10);

                            //__m512 const lower17 = _mm512_permute2f128_ps(dst3, res13, 0x21 ); // 1101 1101 => 0xDD
                            //_m512 const lower7 = _mm512_permute2f128_ps(dst2, res3 , 0x21); // 1101 1101 => 0xDD
                            //__m512 const upper17 = _mm512_blend_ps(dst3, res13, 0xF0); // 1000 1000 => 0x88
                            //__m512 const upper7 = _mm512_blend_ps(dst2, res3, 0xF0); // 1000 1000 => 0x88
                            //dst2 = _mm512_add_ps(lower7, upper7);
                            //dst3 = _mm512_add_ps(lower17, upper17);

                        //__m512 sums2 = _mm512_load_ps(temp); 
                        //__m512 sums2 = _mm512_setzero_ps(); 
                        dst2 = _mm512_add_ps(dst2, upper118);
                        dst3 = _mm512_add_ps(dst3, upper11118);
                        _mm512_store_ps(dst_ptr, dst2);
                        _mm512_store_ps(dst_ptr2, dst3);
                        dst_ptr += simd_ele_width;
                        dst_ptr2 += simd_ele_width;

                       }}} 
#if 0                       
                for (int idx = 0; idx < block_ele_i; ++idx)
                {
                    dst_ptr = dst + (i_outer + i_outer2 + idx)*n + j_outer + j_outer2;
                    memcpy(dst_ptr, packed_dst + idx*block_ele_j, BLOCK_J);
                }
#endif                
                    }
                }
            }
        }
    }
    return NULL;
}

#endif  // __GEMM_H__
