#!/bin/bash

cd ../ && mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=90a
make cutlass_profiler -j48

make test_unit_gemm_device_tensorop_fp8_tma -j
./test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_fp8_tma