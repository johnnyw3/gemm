#include <iostream>
#include <chrono>
#include <x86intrin.h>
#include <stdint.h>
#include <cblas.h>

#define US_PER_S 1000000
#define GIGA     1000000000
#define SIMD_INT_WIDTH 8
#define BLOCK_WIDTH 32 // a 32x32 block of uint32s uses 32K of memory (64K of L1 cache on this CPU)
#define BLOCK_VEC_WIDTH BLOCK_WIDTH / SIMD_INT_WIDTH


void simd_gemm(int *mat1, int *mat2, int *dst, int n);
int read_mat(char *fname, int *n, int **dst);
void print_mat(int *mat, int n);
double get_gflops(std::size_t us, std::size_t n);
void int_convert_and_semm(int *mat1, int *mat2, int *dst, int n);

int main(int argv, char **argc)
{

    int *mat1, *mat2;
    int n;
    read_mat(argc[1], &n, &mat1);
    read_mat(argc[2], &n, &mat2);

    int *dst = (int*)malloc(sizeof(int) * n * n);
    int_convert_and_semm(mat1, mat2, dst, n);
    printf("sizeof int: %d\n", sizeof(int));

    print_mat(dst, n);

    std::size_t time_sum = 0;
    //std::size_t n = 1000;

    auto const start = std::chrono::high_resolution_clock::now();
    /*
    for (std::size_t i = 0; i < n; ++i)
    {
        __m256i v1 = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
        __m256i v2 = _mm256_set_epi32(2, 2, 2, 2, 2, 2, 2, 2);

        __m256i dst = _mm256_mullo_epi32(v1, v2);
    }
    */

    auto const end = std::chrono::high_resolution_clock::now();
    time_sum += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    //uint32_t *res = (uint32_t*) &dst;
    double gflops = get_gflops(time_sum, n*8);
    printf("Time: %zuus, gflops: %f\n", time_sum, gflops);

    return 0;

}

void simd_gemm(int *mat1, int *mat2, int *dst, int n)
{
    int vec_n = n / SIMD_INT_WIDTH;
    int block_n = n / BLOCK_WIDTH; 

    for (int idx_bbx = 0; idx_bbx < block_n; ++ibx_bbx)
    {
        for (int idx_bby = 0; idx_bby < block_n; ++idx_bbx)
        {
            for (int idx_ax = 0; idx_ax < n; ++idx_ax)
            {
                for (int idx_ay = 0; idx_ay < vec_n; ++idx_ay)
                {
                    __m256i a_vec = _mm256_loadu_si256(mat1 + idx_ay*n + idx_ax);

                    for (int idx_bx = 0; idx_bx < BLOCK_VEC_WIDTH; ++idx_bx)
                    {
                        __m256i intermediate_sum = _mm256_setzero_si256();


                        for (int idx_by = 0; idx_by < BLOCK_VEC_WIDTH; ++idx_by)
                        {
                            __m256i = _mm256_loadu_si256(mat2 + (idx_bby*BLOCK_WIDTH + idx_by)*n + (idx_bbx*BLOCK_WIDTH + idx_bx));

                            
                        }
                    }
                }

            }
        }
    }

}

int read_mat(char *fname, int *n, int **dst)
{
    FILE *fp = fopen(fname, "r");
    if (!fp)
    {
        perror("Error opening matrix data file");
        return 1;
    }

    fscanf(fp, "%d", n);
    std::cout << *n**n << "\n";

    *dst = (int*)malloc(sizeof(int) * *n * *n);

    for (int idx = 0; idx < *n**n; ++idx)
    {
        fscanf(fp, "%d", *dst + idx);
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
            printf("%f ", *(mat + idx*n + jdx));
        printf("\n");
    }

}

double get_gflops(std::size_t us, std::size_t n)
{
    double s = us*1.0 / US_PER_S;
    return n / s / GIGA;
}

void int_convert_and_semm(int *mat1, int *mat2, int *dst, int n)
{
    float mat1_fl[n*n], mat2_fl[n*n], dst_fl[n*n];

    for (int idx = 0; idx < n*n; ++idx)
    {
        mat1_fl[idx] = mat1[idx];
        mat2_fl[idx] = mat2[idx];
        dst_fl[idx]  = 0;
    }
    
    print_mat(mat1_fl, n);

    cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                 mat1_fl, n, mat2_fl, n, 1.0, dst_fl, n );


    for (int idx = 0; idx < n*n; ++idx)
    {
        dst[idx] = dst_fl[idx];
    }
    
}
