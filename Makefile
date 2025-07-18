# for logging etc
TEMP_FNAME=$(shell date +%FT%H:%M:%S)_log.txt

NUM_THREADS?=$(shell nproc)

TARGET?=skylake
CXXFLAGS?=-march=$(TARGET) -O3 -flto -g -fsave-optimization-record -DNUM_THREADS=$(NUM_THREADS) -DTARGET=$(TARGET)
CC=clang++

all: bench

build/libgemm.a: gemm.cxx gemm.h simd_common.h
	@mkdir -p build
	$(CC) -o build/gemm.o -c gemm.cxx $(CXXFLAGS)
	ar rcs build/libgemm.a build/gemm.o

bench: build/libgemm.a bench.cxx bench.h
	$(CC) -o build/bench bench.cxx -Lbuild -lgemm -lopenblas $(CXXFLAGS)

.PHONY: clean perf_report
perf_report: bench
	perf stat -d -d -d build/bench mat1_4096.txt mat2_4096.txt 2>&1 | tee $(TEMP_FNAME)
	perf stat -e l2_lines_out.useless_hwpf,cycle_activity.stalls_l1d_miss build/bench mat1_4096.txt mat2_4096.txt 2>&1 | tee -a $(TEMP_FNAME)

clean:
	rm -rf build
