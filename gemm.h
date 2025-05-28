#ifndef __GEMM_H__
#define __GEMM_H__ 1
#include <x86intrin.h>

#define US_PER_S 1000000
#define GIGA     1000000000

#define BLOCK_WIDTH 64 // in bytes -> 128x128 block
                        // a 64x64 block of floats uses 16K of memory (64KB L1d cache on this CPU - i5-8350u)

template<typename T>
void simd_gemm(T *mat1, T *mat2, T *dst, int n);

inline void gemm_inner(float *mat1_ptr, float *mat2_ptr, float *dst_ptr, int simd_ele_width, int block_ele_width);
inline void gemm_inner(double *mat1_ptr, double *mat2_ptr, double *dst_ptr, int simd_ele_width, int block_ele_width);
void cpu_transpose(float *mat, int n);

template<typename T>
void simd_gemm(T *mat1, T *mat2, T *dst, int n)
{
    int simd_ele_width  = SIMD_WIDTH  / sizeof(T);
    int block_ele_width = BLOCK_WIDTH / sizeof(T);
    int vec_n = n / simd_ele_width;

    float *mat1_ptr, *mat2_ptr, *dst_ptr;

    for (int i_outer = 0; i_outer < n; i_outer += block_ele_width)
    {
        for (int j_outer = 0; j_outer < n; j_outer += block_ele_width)
        {
            //T dst_tmp[block_ele_width*block_ele_width];
            for (int k_outer = 0; k_outer < n; k_outer += block_ele_width)
            {
                for (int i_inner = 0; i_inner < block_ele_width; ++i_inner)
                {
                    mat1_ptr = mat1 + (i_outer + i_inner)*n + k_outer;
                    _mm_prefetch(mat1_ptr, _MM_HINT_T0);

                    //dst_ptr = dst_tmp + i_inner*block_ele_width; // + (i_outer + i_inner)*n + j_outer;
                    dst_ptr = dst + (i_outer + i_inner)*n + j_outer;
                    //_mm_prefetch(dst_ptr, _MM_HINT_T0); 
                    
                    for (int j_inner = 0; j_inner < block_ele_width; j_inner += simd_ele_width)
                    {
                        T temp[simd_ele_width]  __attribute__ ((__aligned__(32)));
                        for (int j_inner_inner = 0; j_inner_inner < simd_ele_width; ++j_inner_inner)
                        {
                            mat2_ptr = mat2 + (j_outer + j_inner + j_inner_inner)*n + k_outer;
                            _mm_prefetch(mat2_ptr, _MM_HINT_T0);

                            gemm_inner(mat1_ptr, mat2_ptr, temp + j_inner_inner, simd_ele_width, block_ele_width);
                        }

                        __m256 sums = _mm256_load_ps(temp); 
                        //__m256 sums = _mm256_setzero_ps(); 
                        __m256 dst_vec = _mm256_load_ps(dst_ptr);
                        sums = _mm256_add_ps(sums, dst_vec);
                        _mm256_store_ps(dst_ptr, sums);
                        dst_ptr += simd_ele_width;
                        
                    }
                }
            }
        }
    }

#if 0
    __m256 res_vec1, res_vec2, sums2, sums3; 
    float res[simd_ele_width];
    float *dst_ptr_res;
    dst_ptr = dst;

    for (int idx = 0; idx < n; ++idx)
    {
        dst_ptr     = dst + idx*n*simd_ele_width;
        dst_ptr_res = dst + idx*n;

        for (int jdx = 0; jdx < n; jdx += 2)
        {
            res_vec1 = _mm256_load_ps( dst_ptr + jdx*simd_ele_width);
            res_vec2 = _mm256_load_ps( dst_ptr + jdx*simd_ele_width + simd_ele_width);
            sums2 = _mm256_hadd_ps(res_vec1, res_vec2);
            sums3 = _mm256_hadd_ps(sums2, sums2);
            _mm256_store_ps(res, sums3);

            *( dst_ptr_res + jdx )     = sums3[0] + sums3[4];
            *( dst_ptr_res + jdx + 1 ) = sums3[1] + sums3[5];
        }
    }
#endif
}

inline void gemm_inner(float *mat1_ptr, float *mat2_ptr, float *dst_ptr, int simd_ele_width, int block_ele_width)
{
    __m256 a_vec, b_vec;
    //__m256 sums = _mm256_load_ps(dst_ptr); 
    __m256 sums = _mm256_setzero_ps();

    
    for (int k_inner = 0; k_inner < block_ele_width; k_inner += simd_ele_width)
    {
        a_vec = _mm256_load_ps( mat1_ptr + k_inner );
        b_vec = _mm256_load_ps( mat2_ptr + k_inner );
        sums = _mm256_fmadd_ps(a_vec, b_vec, sums);
    }
    
    
    /*
    __m256 sums2 = _mm256_hadd_ps(sums, sums);
    __m256 sums3 = _mm256_hadd_ps(sums2, sums2);
    float res[simd_ele_width];
    _mm256_store_ps(res, sums3);
    *( dst_ptr ) += res[0] + res[4];
    */
    //int i = _mm256_reduce_add_ps(sums);
    *( dst_ptr ) = _mm256_reduce_add_ps(sums);
    //_mm256_store_ps(dst_ptr, sums);
}

inline void gemm_inner(double *mat1_ptr, double *mat2_ptr, double *dst_ptr, int simd_ele_width, int block_ele_width)
{
    __m256d sums = _mm256_setzero_pd();

    for (int k_inner = 0; k_inner < block_ele_width; k_inner += simd_ele_width)
    {
        __m256d a_vec = _mm256_load_pd( mat1_ptr + k_inner );
        __m256d b_vec = _mm256_load_pd( mat2_ptr + k_inner );
        sums = _mm256_fmadd_pd(a_vec, b_vec, sums);
    }
    
    __m256d sums2 = _mm256_hadd_pd(sums, sums);
    double res[simd_ele_width];
    _mm256_store_pd(res, sums2);
    *( dst_ptr ) += res[0] + res[2];

}

#endif  // __GEMM_H__
