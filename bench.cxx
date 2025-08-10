#include <iostream>
#include <chrono>
#include <stdint.h>
#include <cblas.h>
#include "simd_common.h"
#include "gemm.h"
#include "bench.h"

int main(int argv, char **argc)
{
#ifdef USE_AMX
    printf("Using AMX, %d threads\n", NUM_THREADS);
    __bf16 *mat1, *mat2;
#else
    printf("Using %d-wide SIMD, %d threads\n", SIMD_WIDTH*8, NUM_THREADS);
    float *mat1, *mat2;
#endif

    int n;
    read_mat(argc[1], &n, &mat1);
    read_mat(argc[2], &n, &mat2);
    std::size_t n_large = n;

    std::size_t mat_sz = sizeof(float) * n * n;
    float *dst = (float*)aligned_alloc(64, mat_sz);

    float *dst_cblas = (float*)aligned_alloc(64, (sizeof(float) * n * n));
    std::size_t time_sum_blas =0;
    for (int idx = 0; idx < 10; ++idx)
    {
        memset(dst_cblas, 0, sizeof(float) * n * n);

        auto const start = std::chrono::high_resolution_clock::now();
        cblas_gemm(mat1, mat2, dst_cblas, n);

        auto const end = std::chrono::high_resolution_clock::now();
        time_sum_blas += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    }
    double gflops = get_gflops(time_sum_blas, 10*2*n_large*n_large*n_large);
    printf("OpenBLAS time: %lfs, gflops: %f\n", time_sum_blas/10.0/US_PER_S, gflops);
    //print_mat(dst_cblas, n);

    memset(dst, 0, mat_sz);

    std::size_t time_sum = 0;

#ifndef USE_AMX
    // Transpose only needed for AVX kernels.
    cpu_transpose(mat2, n);
#else
    // For AMX, there is a "relayout" step due to the unconventional structre
    // for the right matrix required by AMX. This is here so we can rerun the
    // relayout procedure for each iteration.
    __bf16 *mat2_orig = (__bf16*)malloc(sizeof(__bf16) * n * n);
    memcpy(mat2_orig, mat2, sizeof(__bf16) * n * n);
#endif

    for (int idx = 0; idx < 10; ++idx)
    {
        auto const start = std::chrono::high_resolution_clock::now();
#ifdef USE_AMX
        amx_relayout(mat2, n, n);
#endif
        simd_gemm(mat1, mat2, dst, n);

        auto const end = std::chrono::high_resolution_clock::now();
        time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //printf("done\n");

        //print_mat(dst, n);
        //print_mat(mat2, n);
        verify_matrix(dst_cblas, dst, n);

        // zero dst so that the algorithm actually needs to successfully run for
        // multiple tests to pass.
        memset(dst, 0, mat_sz);
#ifdef USE_AMX
        // Restore the original matrix so we can rerun the relayout step.
        memcpy(mat2, mat2_orig, sizeof(__bf16) * n * n);
#endif
    }

    printf("\n");

    gflops = get_gflops(time_sum, 10*2*n_large*n_large*n_large);
    printf("Avg time: %lfs, gflops: %f\n", time_sum/10.0/US_PER_S, gflops);
    
    free(mat1);
    free(mat2);
#ifdef USE_AMX
    free(mat2_orig);
#endif
    free(dst);
    free(dst_cblas);
    return 0;

}

int read_mat(char *fname, int *n, __bf16 **dst)
{
    FILE *fp = fopen(fname, "r");
    if (!fp)
    {
        perror("Error opening matrix data file");
        return 1;
    }

    fscanf(fp, "%d", n);
    std::cout << *n << "\n";

    *dst = (__bf16*)aligned_alloc(64, (sizeof(__bf16) * *n * *n));

    for (int idx = 0; idx < *n**n; ++idx)
    {
        float tmp = 0;
        fscanf(fp, "%f", &tmp);
        *(*dst + idx) = (__bf16)tmp;
    }

    fclose(fp);
    return 0;
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

    *dst = (float*)aligned_alloc(64, (sizeof(float) * *n * *n));

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

void print_mat(__bf16 *mat, int n)
{
    for (int idx = 0; idx < n; ++idx)
    {
        for (int jdx = 0; jdx < n; ++jdx)
            printf("%.0f ", (float)*(mat + idx*n + jdx));
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

void cblas_gemm(__bf16 *mat1, __bf16 *mat2, float *dst, int n)
{

    cblas_sbgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                  (bfloat16 *)mat1, n, (bfloat16 *)mat2, n, 1.0, dst, n );

}

void cblas_gemm(float *mat1, float *mat2, float *dst, int n)
{

    cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                 mat1, n, mat2, n, 1.0, dst, n );

}

void verify_matrix(__bf16 *exp, __bf16 *act, int n)
{
    int incorrect = 0;
    for (int idx_x = 0; idx_x < n; ++idx_x)
    {
        for (int idx_y = 0; idx_y < n; ++idx_y)
        {
            __bf16 exp_val = *(exp + idx_y*n + idx_x);
            __bf16 act_val = *(act + idx_y*n + idx_x);

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

void verify_matrix(float *exp, __bf16 *act, int n)
{
    int incorrect = 0;
    for (int idx_x = 0; idx_x < n; ++idx_x)
    {
        for (int idx_y = 0; idx_y < n; ++idx_y)
        {
            float exp_val = *(exp + idx_y*n + idx_x);
            float act_val = (float)*(act + idx_y*n + idx_x);

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
                printf("difference at: (%d, %d). exp: %.2f, act: %.2f\n", idx_x, idx_y, exp_val, act_val);
                incorrect = 1;
            }
        }
    }

    if (!incorrect)
        printf("Matricies are the same\n");
}
