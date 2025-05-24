#include <iostream>
#include <chrono>
#include <x86intrin.h>
#include <stdint.h>
#include <cblas.h>

#define US_PER_S 1000000
#define GIGA     1000000000

#ifdef __AVX512F__
#define SIMD_WIDTH 64   // in bytes -> 512-bit (AVX512)
#elif defined(__AVX2__)
#define SIMD_WIDTH 32   // in bytes -> 256-bit (AVX2)
#else
#define SIMD_WIDTH 16   // in bytes -> 128-bit (original AVX)
#endif

#define BLOCK_WIDTH 512 // in bytes -> 128x128 block
                        // a 64x64 block of floats uses 16K of memory (64KB L1d cache on this CPU - i5-8350u)

// PROTOTYPES
template<typename T>
void simd_gemm(T *mat1, T *mat2, T *dst, int n);

inline void gemm_inner(float *mat1_ptr, float *mat2_ptr, float *dst_ptr, int simd_ele_width, int block_ele_width);
inline void gemm_inner(double *mat1_ptr, double *mat2_ptr, double *dst_ptr, int simd_ele_width, int block_ele_width);
int read_mat(char *fname, int *n, float **dst);
void print_mat(float *mat, int n);
double get_gflops(std::size_t us, std::size_t n);
void cblas_semm(float *mat1, float *mat2, float *dst, int n);
void cpu_transpose(float *mat, int n);
void verify_matrix(float *exp, float *act, int n);

int main(int argv, char **argc)
{
    printf("Using %d-wide SIMD\n", SIMD_WIDTH*8);

    float *mat1, *mat2;
    int n;
    read_mat(argc[1], &n, &mat1);
    read_mat(argc[2], &n, &mat2);

    float *dst_cblas = (float*)aligned_alloc(32, (sizeof(float) * n * n));
    cblas_semm(mat1, mat2, dst_cblas, n);
    //print_mat(dst_cblas, n);

    float *dst = (float*)malloc(sizeof(float) * n * n);
    for (int idx = 0; idx < n; ++idx)
        for (int jdx = 0; jdx < n; ++jdx)
            *(dst + idx*n + jdx) = 0;

    std::size_t time_sum = 0;

    cpu_transpose(mat2, n);

    for (int idx = 0; idx < 10; ++idx)
    {
        auto const start = std::chrono::high_resolution_clock::now();
        simd_gemm(mat1, mat2, dst, n);

        auto const end = std::chrono::high_resolution_clock::now();
        time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        //print_mat(dst, n);
        verify_matrix(dst_cblas, dst, n);

        for (int idx = 0; idx < n; ++idx)
            for (int jdx = 0; jdx < n; ++jdx)
                *(dst + idx*n + jdx) = 0;
    }

    printf("\n");

    std::size_t n_large = n;
    double gflops = get_gflops(time_sum, 10*2*n_large*n_large*n_large);
    printf("Avg time: %zuus, gflops: %f\n", time_sum/10, gflops);
    
    free(mat1);
    free(mat2);
    free(dst);
    return 0;

}

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
            for (int k_outer = 0; k_outer < n; k_outer += block_ele_width)
            {
                for (int i_inner = 0; i_inner < block_ele_width; ++i_inner)
                {
                    mat1_ptr = mat1 + (i_outer + i_inner)*n + k_outer;
                    _mm_prefetch(mat1_ptr, _MM_HINT_T0);

                    dst_ptr = dst + (i_outer + i_inner)*n + j_outer;
                    _mm_prefetch(dst_ptr, _MM_HINT_T0); 
                    
                    for (int j_inner = 0; j_inner < block_ele_width; ++j_inner)
                    {
                        mat2_ptr = mat2 + (j_outer + j_inner)*n + k_outer;
                        _mm_prefetch(mat2_ptr, _MM_HINT_T0);

                        gemm_inner(mat1_ptr, mat2_ptr, dst_ptr + j_inner, simd_ele_width, block_ele_width);
                    }
                }
            }
        }
    }
}


inline void gemm_inner(float *mat1_ptr, float *mat2_ptr, float *dst_ptr, int simd_ele_width, int block_ele_width)
{
    __m256 a_vec, b_vec;
    __m256 sums = _mm256_setzero_ps();

    for (int k_inner = 0; k_inner < block_ele_width; k_inner += simd_ele_width)
    {
        a_vec = _mm256_load_ps( mat1_ptr + k_inner );
        b_vec = _mm256_load_ps( mat2_ptr + k_inner );
        sums = _mm256_fmadd_ps(a_vec, b_vec, sums);
    }
    
    __m256 sums2 = _mm256_hadd_ps(sums, sums);
    __m256 sums3 = _mm256_hadd_ps(sums2, sums2);
    float res[simd_ele_width];
    _mm256_store_ps(res, sums3);
    *( dst_ptr ) += res[0] + res[4];

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

int read_mat(char *fname, int *n, float **dst)
{
    FILE *fp = fopen(fname, "r");
    if (!fp)
    {
        perror("Error opening matrix data file");
        return 1;
    }

    fscanf(fp, "%d", n);
    std::cout << *n << "\n";

    *dst = (float*)aligned_alloc(32, (sizeof(float) * *n * *n));

    for (int idx = 0; idx < *n**n; ++idx)
    {
        fscanf(fp, "%f", *dst + idx);
    }

    fclose(fp);
    return 0;
}

void print_mat(int *mat, int n)
{
    for (int idx = 0; idx < n; ++idx)
    {
        for (int jdx = 0; jdx < n; ++jdx)
            printf("%d ", *(mat + idx*n + jdx));
        printf("\n");
    }

}

void print_mat(float *mat, int n)
{
    for (int idx = 0; idx < n; ++idx)
    {
        for (int jdx = 0; jdx < n; ++jdx)
            printf("%.0f ", *(mat + idx*n + jdx));
        printf("\n");
    }

}

double get_gflops(std::size_t us, std::size_t n)
{
    double s = us*1.0 / US_PER_S;
    return n / s / GIGA;
}

void cblas_semm(float *mat1, float *mat2, float *dst, int n)
{
    //float dst_fl[n*n];

    //for (int idx = 0; idx < n*n; ++idx)
    //{
    //    dst_fl[idx]  = 0;
    //}
    
    //print_mat(mat2_fl, n);

    cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                 mat1, n, mat2, n, 1.0, dst, n );


    //for (int idx = 0; idx < n*n; ++idx)
    //{
    //    dst[idx] = dst_fl[idx];
    //}
    
}

void verify_matrix(float *exp, float *act, int n)
{
    int incorrect = 0;
    for (int idx_x = 0; idx_x < n; ++idx_x)
    {
        for (int idx_y = 0; idx_y < n; ++idx_y)
        {
            float exp_val = *(exp + idx_y*n + idx_x);
            float act_val = *(act + idx_y*n + idx_x);

            if (exp_val != act_val)
            {
                printf("difference at: (%d, %d). exp: %f, act: %f\n", idx_x, idx_y, exp_val, act_val);
                incorrect = 1;
            }
        }
    }

    if (!incorrect)
        printf("Matricies are the same\n");
}

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
