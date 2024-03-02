import ctypes
import sys
import torch

from ctypes import byref, c_size_t, c_void_p

if __name__ == "__main__":
    torch.cuda.init()
    path = sys.argv[1]
    lib = ctypes.cdll.LoadLibrary(path)

    X = torch.rand(1000, 16).cuda().half()
    W = torch.rand(1000, 16).cuda().half()
    Y = torch.zeros(1000, 1000).cuda().half()

    stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)

    ret = lib.cuda_cutlass_gemm(
        c_void_p(X.data_ptr()),
        c_void_p(W.data_ptr()),
        c_void_p(Y.data_ptr()),
        None,
        None,
        stream_ptr,
    )

    torch.cuda.synchronize()


    print(f"{ret=}")

    Y1 = torch.nn.functional.linear(X, W)

    torch.testing.assert_close(Y, Y1)