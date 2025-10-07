# for logging etc
TEMP_FNAME=$(shell date +%FT%H:%M:%S)_log.txt

NUM_THREADS?=$(shell nproc)

TARGET?=skylake
CXXFLAGS?=-march=$(TARGET) -O3 -flto -g -fsave-optimization-record -DNUM_THREADS=$(NUM_THREADS) -DTARGET=$(TARGET)
USE_MKL?=no
CC=clang++
INSTALL?=install

all: build/bench

build/libgemm.a: gemm.cxx gemm.h simd_common.h
	@mkdir -p build
	$(CC) -o build/gemm.o -c gemm.cxx $(CXXFLAGS)
	ar rcs build/libgemm.a build/gemm.o

build/bench: build/libgemm.a bench.cxx bench.h
ifeq ($(USE_MKL),yes)
	@#$(CC) -o build/bench bench.cxx -fopenmp -Lbuild -lgemm -m64  -I"${MKLROOT}/include" -L${MKLROOT}/lib -lmkl -Wl,--no-as-needed -lpthread -lm -ldl $(CXXFLAGS)
	$(CC) -o build/bench bench.cxx -fopenmp -Lbuild $(LDFLAGS) -lgemm -m64 -I"/usr/include/mkl"  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -DUSE_MKL $(CXXFLAGS)
else
	@#-I../OpenBLAS/install/include -L../OpenBLAS/install/lib
	$(CC) -o build/bench bench.cxx -Lbuild $(LDFLAGS) -lgemm -lopenblas $(CXXFLAGS)
endif

install: build/libgemm.a
	$(INSTALL) -CD build/libgemm.a $(PREFIX)/usr/lib/gemm/libgemm.a
	$(INSTALL) -CD -m 644 gemm.h $(PREFIX)/usr/include/gemm/gemm.h

.PHONY: clean perf_report install
perf_report: bench
	perf stat -d -d -d build/bench mat1_4096.txt mat2_4096.txt 2>&1 | tee $(TEMP_FNAME)
	perf stat -e l2_lines_out.useless_hwpf,cycle_activity.stalls_l1d_miss build/bench mat1_4096.txt mat2_4096.txt 2>&1 | tee -a $(TEMP_FNAME)

clean:
	rm -rf build
