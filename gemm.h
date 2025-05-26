#ifndef __GEMM_H__
#define __GEMM_H__ 1

#define US_PER_S 1000000
#define GIGA     1000000000

#define BLOCK_WIDTH 512 // in bytes -> 128x128 block
                        // a 64x64 block of floats uses 16K of memory (64KB L1d cache on this CPU - i5-8350u)

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

#endif  // __GEMM_H__
