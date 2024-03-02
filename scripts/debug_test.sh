#!/bin/bash

nvcc -t=0 -lineinfo -g -G \
--threads 16 \
-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_DEBUG_TRACE_LEVEL=1 \
-w -gencode=arch=compute_90a,code=[sm_90a,compute_90a] \
-O1 -std=c++17 --expt-relaxed-constexpr \
-Xcompiler=-fPIC -Xcompiler=-fno-strict-aliasing -Xcompiler -fvisibility=hidden \
-Xcompiler=-Wconversion \
-I/home/fhu/git/cutlass/include -I/home/fhu/git/cutlass/tools/library/include -I/home/fhu/git/cutlass/tools/library/src -I/home/fhu/git/cutlass/tools/util/include \
-L/usr/local/cuda-12.1/lib64 -L/usr/local/cuda-12.1/lib64/stubs -lcuda -lcudart \
-shared -o test.so test.cu
