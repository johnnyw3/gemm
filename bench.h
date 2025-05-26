#ifndef __BENCH_H__
#define __BENCH_H__

int read_mat(char *fname, int *n, float **dst);
void print_mat(float *mat, int n);
double get_gflops(std::size_t us, std::size_t n);
void cblas_semm(float *mat1, float *mat2, float *dst, int n);
void verify_matrix(float *exp, float *act, int n);

#endif // __BENCH_H__
