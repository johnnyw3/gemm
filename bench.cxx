#include <iostream>
#include <chrono>
#include <stdint.h>
#include <cblas.h>
#include "simd_common.h"
#include "gemm.h"
#include "bench.h"

int main(int argv, char **argc)
{
    printf("Using %d-wide SIMD, %d threads\n", SIMD_WIDTH*8, NUM_THREADS);

    float *mat1, *mat2;
    int n;
    read_mat(argc[1], &n, &mat1);
    read_mat(argc[2], &n, &mat2);
    std::size_t n_large = n;

    float *dst_cblas = (float*)aligned_alloc(64, (sizeof(float) * n * n));
    std::size_t time_sum_blas =0;
    for (int idx = 0; idx < 5; ++idx)
    {
        for (int idx = 0; idx < n; ++idx)
            for (int jdx = 0; jdx < n; ++jdx)
                *(dst_cblas + idx*n + jdx) = 0;

        auto const start = std::chrono::high_resolution_clock::now();
        cblas_semm(mat1, mat2, dst_cblas, n);

        auto const end = std::chrono::high_resolution_clock::now();
        time_sum_blas += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    }
    double gflops = get_gflops(time_sum_blas, 5*2*n_large*n_large*n_large);
    printf("OpenBLAS time: %lfs, gflops: %f\n", time_sum_blas/5.0/US_PER_S, gflops);
    //print_mat(dst_cblas, n);

    float *dst = (float*)aligned_alloc(64, 8 * sizeof(float) * n * n);
    for (int idx = 0; idx < n; ++idx)
        for (int jdx = 0; jdx < n; ++jdx)
            *(dst + idx*n + jdx) = 0;

    std::size_t time_sum = 0;

    cpu_transpose(mat2, n);
    //cpu_transpose(mat1, n);
    //cpu_transpose(dst, n);

    for (int idx = 0; idx < 5; ++idx)
    {
        auto const start = std::chrono::high_resolution_clock::now();
        simd_gemm(mat1, mat2, dst, n);

        auto const end = std::chrono::high_resolution_clock::now();
        time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //printf("done\n");

        //print_mat(dst, n);
        verify_matrix(dst_cblas, dst, n);

        for (int idx = 0; idx < n; ++idx)
            for (int jdx = 0; jdx < n; ++jdx)
                *(dst + idx*n + jdx) = 0;
    }

    printf("\n");

    gflops = get_gflops(time_sum, 5*2*n_large*n_large*n_large);
    printf("Avg time: %lfs, gflops: %f\n", time_sum/5.0/US_PER_S, gflops);
    
    free(mat1);
    free(mat2);
    free(dst);
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
