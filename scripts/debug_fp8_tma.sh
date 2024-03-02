#!/bin/bash

cd ../build/
cmake -DCUTLASS_NVCC_ARCHS=90a -DCMAKE_BUILD_TYPE=Debug -DCUTLASS_NVCC_KEEP:STRING=ON -DCUTLASS_ENABLE_F16C:STRING=ON -DCUTLASS_NVCC_EMBED_PTX:STRING=ON ..
make test_unit_gemm_device_tensorop_fp8_tma -j

ncu -f -o cutlass_fp8_tma_kernel \
  --import-source yes \
  --clock-control none \
  --set full \
  --print-summary per-kernel \
  --launch-count 3 \
 ./test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_fp8_tma --gtest_filter=*fast_accum*

mv cutlass_fp8_tma_kernel.ncu-rep ../scripts

cd ../scripts


# https://github.com/NVIDIA/cutlass/issues/1044

# nvcc -t=0 -lineinfo -g -G \
# -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_DEBUG_TRACE_LEVEL=1 \
# -w -gencode=arch=compute_90a,code=[sm_90a,compute_90a] \
# -O1 -std=c++17 --expt-relaxed-constexpr \
# -Xcompiler=-fPIC -Xcompiler=-fno-strict-aliasing -Xcompiler -fvisibility=hidden \
# -Xcompiler=-Wconversion \
# -I/home/yingz/pytorch/torch/../third_party/cutlass/include -I/home/yingz/pytorch/torch/../third_party/cutlass/tools/library/include -I/home/yingz/pytorch/torch/../third_party/cutlass/tools/library/src -I/home/yingz/pytorch/torch/../third_party/cutlass/tools/util/include \
# -L/usr/local/cuda-12.1/lib64 -L/usr/local/cuda-12.1/lib64/stubs -lcuda -lcudart \
# -shared -o test.so test.cu

# /usr/bin/cmake --no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING=Debug -DCUTLASS_NVCC_ARCHS:STRING=90a -DCUTLASS_NVCC_KEEP:STRING=ON -DCUTLASS_ENABLE_F16C:STRING=ON -DCUTLASS_LIBRARY_KERNELS:STRING=cutlass3x_sm90_tensorop_s64x128x16gemm_f16_f16*128x128x64*2x1x1_0*tnn*align8 -DCUTLASS_LIBRARY_IGNORE_KERNELS:STRING=gemm_grouped*,gemm_planar* -DCUTLASS_ENABLE_CUBLAS:STRING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE 