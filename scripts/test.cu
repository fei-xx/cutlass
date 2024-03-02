

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"



// We compile all models with -fvisibility=hidden. Any symbols that need to be
// exposed in the final shared library must be declared with PT_EXPORT to make
// them visible.
#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define PT_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define PT_EXPORT __declspec(dllexport)
#else
#define PT_EXPORT
#endif
#endif

using bfloat16 = nv_bfloat16;

#define CUTLASS_CHECK(status)                                                      \
{                                                                                  \
  cutlass::Status error = status;                                                  \
  if (error != cutlass::Status::kSuccess) {                                        \
    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \
        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \
    std::cerr << msg << std::endl;                                                 \
    throw std::runtime_error(msg);                                                 \
  }                                                                                \
}




using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::half_t, cutlass::half_t,
    void, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      sizeof(typename cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_epilogue::SharedStorage)>,
  cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8
using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_mainloop,
    cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_epilogue>;

// Define named type
struct cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8 :
  public cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_base { };


  using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_device_type = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8>;


// When workspace_size is not a nullptr, populates requested workspace_size and returns.
// Otherwise, compuates the Gemm kernel using the given workspace ptr.
extern "C" {
PT_EXPORT int cuda_cutlass_gemm(const half* X, const half* W, half* Y, size_t* workspace_size, uint8_t* workspace, cudaStream_t stream) {
  try {
  printf("X: %p\n", (void*)(X));
  printf("W: %p\n", (void*)(W));
  // printf("Bias: %p\n", (void*)(Bias));
  printf("Y: %p\n", (void*)(Y));
  printf("workspace size: %p\n", (void*)(workspace_size));
  printf("workspace: %p\n", (void*)(workspace));
  printf("stream: %p\n", (void*)(stream));

  
  {
    if (!X) {
      int64_t X_size = 16000L;
      if (X_size > 0) {
        throw std::runtime_error("input X is null!");
      }
    }
  }

  
  {
    if (!W) {
      int64_t W_size = 16000L;
      if (W_size > 0) {
        throw std::runtime_error("input W is null!");
      }
    }
  }

  
  
  {
    if (!Y) {
      int64_t Y_size = 1000000L;
      if (Y_size > 0) {
        throw std::runtime_error("input Y is null!");
      }
    }
  }


  int64_t B = 1;
  int64_t M = 1000L;
  int64_t K = 16L;
  int64_t N = 1000L;

  using ElementComputeEpilogue = cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_device_type::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_device_type::Arguments arguments;
  
  // Initialize GemmUniversal3xInstance arguments.
  arguments = {
    cutlass::gemm::GemmUniversalMode::kGemm,  // GemmUniversalMode mode
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K),
      static_cast<coord_t>(B)
    }, // ProblemShape problem_shape
    {
      (cutlass::half_t*)(X),  // ElementA const* ptr_A
      { 16L /* stride_x0 */, cute::Int<1>{} /* stride_x1 */, 0 /* batch_stride_x */},  // StrideA dA
      (cutlass::half_t*)(W),  // ElementB const* ptr_B
      { 16L /* stride_w1 */, cute::Int<1>{} /* stride_w0 */, 0 /* batch_stride_w */},  // StrideB dB
    },  // MainloopArguments mainloop
    
    {
      {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename ThreadEpilogueOp::Params thread
      nullptr,  // ElementC const* ptr_C
      { cute::Int<1>{} /* stride_bias0 */, cute::Int<1>{} /* stride_bias1 */, 0 /* batch_stride_bias */},  // StrideC dC
      (cutlass::half_t*)(Y),  // ElementD const* ptr_D
      { 1000L /* stride_y0 */, cute::Int<1>{} /* stride_y1 */, 0 /* batch_stride_y */},  // StrideD dD
    },  // EpilogueArguments epilogue, no TMA
  };

  cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_f16_f16_128x128x64_2x1x1_0_tnt_align8_device_type gemm_op;

  if (workspace_size) {
    *workspace_size = gemm_op.get_workspace_size(arguments);
    return 0;
  }

  {
    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
  }
  {
    auto status = gemm_op(stream);
    CUTLASS_CHECK(status);
  }
  }
  catch (...) {
    return -1;
  }

  return 0;
}
}