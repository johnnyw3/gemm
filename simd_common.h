#ifndef __SIMD_COMMON_H__
#define __SIMD_COMMON_H__
#include <x86intrin.h>

#ifdef __AMX_TILE__
#define USE_AMX
#define SIMD_WIDTH  64 // AMX tiles are at most 64 bytes wide
#define SIMD_HEIGHT 16
#elif defined(__AVX512F__)
#define SIMD_WIDTH  64 // in bytes -> 512-bit (AVX512)
#define SIMD_HEIGHT  1
#define USE_AVX512
#elif defined(__AVX2__)
#define SIMD_WIDTH  32 // in bytes -> 256-bit (AVX2)
#define SIMD_HEIGHT  1
#else
// NOTE: AVX (original) support not yet implemented
#define SIMD_WIDTH  16 // in bytes -> 128-bit (original AVX)
#define SIMD_HEIGHT  1
#error "Original (128-bit) AVX support not implemented."
#endif

#ifdef USE_AMX
#define BLOCK_I 2048
#define BLOCK_J 1024
#if NUM_THREADS == 1
#define BLOCK_K 1024
#else
#define BLOCK_K 1024
#endif
#define SBLOCK_I 4096
#define SBLOCK_J 2048
#define SBLOCK_K 2048
// code optimized for the larger cache sizes on Tiger Lake
#elif defined(USE_AVX512)
#define BLOCK_I 64
#define BLOCK_J 128
#if NUM_THREADS == 1
#define BLOCK_K 1024
#else
#define BLOCK_K 512
#endif
#define SBLOCK_I 4096
#define SBLOCK_J 2048
#define SBLOCK_K 1024

#else
// default block sizes (in bytes), optimized for Skylake-sizes caches
#define BLOCK_I 64 // in bytes -> 128x128 block
                    // a 64x64 block of floats uses 16K of memory (64KB L1d cache on this CPU - i5-8350u)
#define BLOCK_J 64 
#define BLOCK_K 1024
#define SBLOCK_I 2048 
#define SBLOCK_J 2048 
#define SBLOCK_K 2048 
#endif


// adapted from GCC's _mm512_reduce_add_ps macro for use on AVX2
inline int _mm256_reduce_add_ps(__m256 vec)
{
    __m128 lower = _mm256_extractf128_ps(vec, 0);
    __m128 upper = _mm256_extractf128_ps(vec, 1);
    __m128 res   = _mm_add_ps(lower, upper);
    __m128 shuf  = _mm_permute_ps(res, 0x1B); // {2, 3, 0, 1}
    res          = _mm_add_ps(res, shuf);
    return res[0] + res[1];
}

#endif // __SIMD_COMMON_H__
