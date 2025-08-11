# GEMM 

High performance GEMM kernels written in C++ using AVX and AMX
intrinsics. Currently achieves ~80% performance of OpenBLAS (at least on my
systems). 

Some optimization techniques were inspired by Salykov's article on the topic [2],
but the algorithm used here is different than the one described in the article.
On my systems, I can achieve comparable performance to Salykov's code.

# Benchmarks

Metrics are in GFLOPs; speedups are compared to OpenBLAS. Types are `fp32` for
AVX-based kernels and `bf16` input/`fp32` result for AMX.

**Single-threaded, n=4096** average of 10 runs

| Kernel | CPU | This algorithm | OpenBLAS | Speedup |
|:-------|:----|---------------:|---------:|:------------------|
AVX2 | **Skylake (Kaby Lake)** i5-8350u | 76 | 100 | 0.76 |
AVX-512 | **Tiger Lake** i5-1135G7 | 106 | 122 | 0.87 |
AMX | **Emerald Rapids** Xeon 6972P | 1090 | 1170 | 0.93 |

**4 threads, n=4096** average of 10 runs

| Kernel | CPU | This algorithm | OpenBLAS | Speedup |
|:-------|:----|---------------:|---------:|:------------------|
AVX2 | **Skylake (Kaby Lake)** i5-8350u | 213 | 274 | 0.78 |
AVX-512 | **Tiger Lake** i5-1135G7 | 380 | 465 | 0.82 |

# Prerequisites

## Hardware

A modern x86 processor supporting AVX2, AVX-512, or AMX must be used.

## Software

For the library:

* `pthreads` support (for multithreading)
* A version of `make` that supports the `shell` directive
* A modern C++ compiler. For AMX support, `clang` version 16+ or `gcc` version
  13+ must be used as these compilers support the `__bf16` type.

For the `bench` program:

* OpenBLAS (for verifying correctness and providing a baseline to compare against)
    - If using AMX, OpenBLAS must be compiled with `BUILD_BFLOAT16=1` (NOT the
      default)

# Build

To build, use the given makefile, specifying your march. For example:

```bash
$ make TARGET=skylake
```

The fastest available kernel (AVX2, AVX-512, or AMX) will be chosen based on the
features available in the `TARGET` architecture.

By default, multithreading is enabled using as many threads as logical
processors on your system. To change the number of threads, pass the
`NUM_THREADS=<num>` option:

```bash
$ make TARGET=skylake NUM_THREADS=4
```

You'll need to `make clean` first before changing configurations.

# References

[1] U. Drepper, “What Every Programmer Should Know About Memory,” Nov. 2007, [Online]. Available: https://people.freebsd.org/~lstewart/articles/cpumemory.pdf

[2] A. Salykov, “Advanced Matrix Multiplication Optimization on Modern Multi-Core Processors,” salykova. Accessed: Jun. 20, 2025. [Online]. Available: https://salykova.github.io/matmul-cpu

[3] “Intel Advanced Vector Extensions 512  (Intel AVX-512) - Permuting Data Within  and Between AVX Registers,” Intel. Accessed: Jun. 19, 2025. [Online]. Available: https://builders.intel.com/docs/networkbuilders/intel-avx-512-permuting-data-within-and-between-avx-registers-technology-guide-1668169807.pdf
