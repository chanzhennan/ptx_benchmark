#include "../op.h"

using namespace std;

#define CUDA_CHECK(ans) \
  { cuda_assert((ans), __FILE__, __LINE__); }

inline void cuda_assert(CUresult code, const char* file, int line) {
  if (code != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(code, &err_str);
    fprintf(stderr, "CUDA Error: %s %s %d\n", err_str, file, line);
    exit(code);
  }
}

void AddPi() {
  int result[4];

  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction function;
  char moduleFile[] =
      "/share/chenzhennan/sourceCode/cuDa/ptx_benchmark/gpumark/kernels/"
      "addpi.ptx";
  char kernelName[] = "AddPi";

  CUresult r = CUDA_SUCCESS;

  CUresult err = cuInit(0);
  cuDeviceGet(&device, 0);
  cuCtxCreate(&context, 0, device);

  CUDA_CHECK(cuModuleLoad(&module, moduleFile));
  CUDA_CHECK(cuModuleGetFunction(&function, module, kernelName));

  int size = 4;
  unsigned int byteSize = size * sizeof(int);
  int* h_a = (int*)malloc(byteSize);
  CUdeviceptr d_a;
  cuMemAlloc(&d_a, byteSize);

  for (int i = 0; i < size; i++) h_a[i] = i;

  cuMemcpyHtoD(d_a, h_a, byteSize);

  int64_t a = 10;

  void* args[]{&a};
  // cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int
  // gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int
  // blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream
  // hStream, void** kernelParams, void** extra )
  cuLaunchKernel(function, 1, 1, 1, size, 1, 1, 0, 0, (void**)args, 0);

  cuCtxSynchronize();
}
