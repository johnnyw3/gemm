#ifndef __SIMD_COMMON_H__
#define __SIMD_COMMON_H__

#ifdef __AVX512F__
#define SIMD_WIDTH 64   // in bytes -> 512-bit (AVX512)
#elif defined(__AVX2__)
#define SIMD_WIDTH 32   // in bytes -> 256-bit (AVX2)
#else
#define SIMD_WIDTH 16   // in bytes -> 128-bit (original AVX)
#endif

#endif // __SIMD_COMMON_H__
