#include <iostream>
#include <chrono>
#include <x86intrin.h>
#include <stdint.h>
#include <cblas.h>

#define US_PER_S 1000000
#define GIGA     1000000000
#define SIMD_WIDTH 8
#define BLOCK_WIDTH 64 // a 64x64 block of floats uses 16K of memory (64KB L1d cache on this CPU - i5-8350u)
#define BLOCK_VEC_WIDTH BLOCK_WIDTH / SIMD_WIDTH


void simd_gemm(float *mat1, float *mat2, float *dst, int n);
int read_mat(char *fname, int *n, float **dst);
void print_mat(float *mat, int n);
double get_gflops(std::size_t us, std::size_t n);
void cblas_semm(float *mat1, float *mat2, float *dst, int n);
void cpu_transpose(float *mat, int n);
void verify_matrix(float *exp, float *act, int n);

int main(int argv, char **argc)
{

    float *mat1, *mat2;
    int n;
    read_mat(argc[1], &n, &mat1);
    read_mat(argc[2], &n, &mat2);

    float *dst_cblas = (float*)aligned_alloc(32, (sizeof(float) * n * n));
    cblas_semm(mat1, mat2, dst_cblas, n);

    //print_mat(dst_cblas, n);
    //free(dst);
    float *dst = (float*)malloc(sizeof(float) * n * n);
    for (int idx = 0; idx < n; ++idx)
        for (int jdx = 0; jdx < n; ++jdx)
            *(dst + idx*n + jdx) = 0;

    std::size_t time_sum = 0;
    //std::size_t n = 1000;

    cpu_transpose(mat2, n);
    //print_mat(mat2, n);
    auto const start = std::chrono::high_resolution_clock::now();
    simd_gemm(mat1, mat2, dst, n);

    auto const end = std::chrono::high_resolution_clock::now();
    time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    printf("\n");
    //print_mat(dst, n);
    //uint32_t *res = (uint32_t*) &dst;
    std::size_t n_large = n;
    std::cout << 2*n_large*n_large*n_large << "\n";
    double gflops = get_gflops(time_sum, 2*n_large*n_large*n_large);
    printf("Time: %zuus, gflops: %f\n", time_sum, gflops);
    
    verify_matrix(dst_cblas, dst, n);

    free(mat1);
    free(mat2);
    free(dst);
    return 0;

}

void simd_gemm(float *mat1, float *mat2, float *dst, int n)
{
    int vec_n = n / SIMD_WIDTH;
    int block_n = n / BLOCK_WIDTH; 

    float *mat1_ptr, *mat2_ptr, *dst_ptr;
//#if 0
    for (int i_outer = 0; i_outer < n; i_outer += BLOCK_WIDTH)
    {
        for (int j_outer = 0; j_outer < n; j_outer += BLOCK_WIDTH)
        {
            for (int k_outer = 0; k_outer < n; k_outer += BLOCK_WIDTH)
            {
                for (int i_inner = 0; i_inner < BLOCK_WIDTH; ++i_inner)
                {
                    mat1_ptr = mat1 + (i_outer + i_inner)*n + k_outer;
                    _mm_prefetch(mat1_ptr, _MM_HINT_T0);

                    dst_ptr = dst + (i_outer + i_inner)*n + j_outer;
                    _mm_prefetch(dst_ptr, _MM_HINT_T0); 
                    
                    for (int j_inner = 0; j_inner < BLOCK_WIDTH; ++j_inner)
                    {
                        mat2_ptr = mat2 + (j_outer + j_inner)*n + k_outer;
                        _mm_prefetch(mat2_ptr, _MM_HINT_T0);

                        __m256 sums = _mm256_setzero_ps();

                        for (int k_inner = 0; k_inner < BLOCK_WIDTH; k_inner += SIMD_WIDTH)
                        {
                            __m256 a_vec = _mm256_load_ps( mat1_ptr + k_inner );
                            __m256 b_vec = _mm256_load_ps( mat2_ptr + k_inner );
                            sums = _mm256_fmadd_ps(a_vec, b_vec, sums);
                        }
                        
                        __m256 sums2 = _mm256_hadd_ps(sums, sums);
                        __m256 sums3 = _mm256_hadd_ps(sums2, sums2);
                        float res[8];
                        _mm256_store_ps(res, sums3);
                        *( dst_ptr + j_inner ) += res[0] + res[4];
                        
                    }
                }
            }
        }
    }
//#endif
#if 0
    for (int idx_bbx = 0; idx_bbx < block_n; ++idx_bbx)
    {
        for (int idx_bby = 0; idx_bby < block_n; ++idx_bby)
        {
            //for (int idx_ax = 0; idx_ax < BLOCK_WIDTH; idx_ax += 8)
            //{
                for (int idx_ay = 0; idx_ay < n; ++idx_ay)
                {
                    //__m256 a_vec = _mm256_set1_ps(mat1 + idx_ay*n + idx_ax);
                    for (int idx_by = 0; idx_by < BLOCK_WIDTH; ++idx_by)
                    {
                        __m256 sums = _mm256_setzero_ps();
                        //__m256 sums = _mm256_loadu_ps(dst + idx_ay*n + (idx_bbx*BLOCK_WIDTH + idx_bx));

                        for (int idx_bx = 0; idx_bx < BLOCK_WIDTH; idx_bx += 8)
                        {
                            __m256 a_vec = _mm256_load_ps(mat1 + idx_ay*n + (idx_bbx*BLOCK_WIDTH + idx_bx));
                            __m256 b_vec = _mm256_load_ps(mat2 + (idx_bby*BLOCK_WIDTH + idx_by)*n + (idx_bbx*BLOCK_WIDTH + idx_bx));
                            sums = _mm256_fmadd_ps(a_vec, b_vec, sums);

                        }
                        //sums = _mm256_set1_ps(1);
                        //float res[8];
                        //_mm256_storeu_ps(dst + idx_ay*n + (idx_bbx*BLOCK_WIDTH + idx_bx), sums);
                        //_mm256_storeu_ps(res, sums);
                        //float sum = 0;
                        //for (int idx = 0; idx < 8; ++idx)
                        //   sum += res[idx];
                        __m256 sums2 = _mm256_hadd_ps(sums, sums);
                        __m256 sums3 = _mm256_hadd_ps(sums2, sums2);
                        float res[8];
                        _mm256_store_ps(res, sums3);
                        *(dst + idx_ay*n + (idx_bby*BLOCK_WIDTH + idx_by)) += res[0] + res[4];
                    }
                }
            //}
        }
    }
#endif
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
