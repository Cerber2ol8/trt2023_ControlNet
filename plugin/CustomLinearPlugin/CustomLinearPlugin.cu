#include "CustomLinearPlugin.h"
#include <numeric>
using namespace nvinfer1;

PluginFieldCollection linearPluginCreator::fc_{};
std::vector<PluginField> linearPluginCreator::attr_;

constexpr int kWarpSize = 32;


template <typename T>
__global__ void CustomLinearKernel(const T *__restrict__ pInput, // input x 1, 77, 320
                                const T *__restrict__ gamma,  // weight 320
                                const T *__restrict__ beta, // bias  320
                                T *__restrict__ pOutput, const int blockSize, const int lenData) {
  const int thIdx = threadIdx.x;
  const int idx = blockIdx.x * blockSize + threadIdx.x;


  if (idx < lenData) {
    pOutput[idx] = gamma[thIdx] * pInput[idx] + beta[thIdx];
  }
  __syncwarp();

}


template <typename T>
__global__ void CustomLinearKernel(const T *__restrict__ pInput, // input x 1, 77, 320
                                const T *__restrict__ gamma,  // weight 320
                                const T *__restrict__ beta, // bias  320
                                T *__restrict__ pOutput, const int blockSize, const int layerDim, const int lenData) {
  const int thIdx = threadIdx.x;
  const int idx = blockIdx.x * blockSize + threadIdx.x;
  const int n = idx % layerDim;

  if (idx < lenData) {
    pOutput[idx] = gamma[n] * pInput[idx] + beta[n];
  }
  __syncwarp();

}

int64_t volume(nvinfer1::Dims const &d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}
int32_t linearPlugin::enqueue(const PluginTensorDesc *inputDesc,
                                 const PluginTensorDesc *outputDesc,
                                 const void *const *inputs,
                                 void *const *outputs, void *workspace,
                                 cudaStream_t stream) noexcept {



  // const int blocksize = ((n_ - 1) / 32 + 1) * 32;
  
  // n = 320 or 640 or 1280
  // const int blocksize = 32;
  
  const int64_t len = volume(inputDesc[0].dims);  

  if( n_ < 1024)
  {
    const int blocksize = n_;
    const int bolck_num = len / blocksize;

    if (inputDesc[0].type == DataType::kFLOAT) {
      CustomLinearKernel<<<bolck_num, blocksize, 0, stream>>>(
      (float *)inputs[0], weight_gpu_, bias_gpu_, (float *)outputs[0], blocksize, len);

    } else if (
      inputDesc[0].type == DataType::kHALF) {
      CustomLinearKernel<<<bolck_num, blocksize, 0, stream>>>(
      (__half *)inputs[0], weight_half_gpu_, bias_half_gpu_,
      (__half *)outputs[0], blocksize, len);

    } else {
      printf("Unsupport datatype!\n");
    }

  } else {

    const int blocksize = 64;
    const int bolck_num = len / blocksize;

    if (inputDesc[0].type == DataType::kFLOAT) {

      CustomLinearKernel<<<bolck_num, blocksize, 0, stream>>>(
      (float *)inputs[0], weight_gpu_, bias_gpu_, (float *)outputs[0], blocksize, n_, len);

    } else if (inputDesc[0].type == DataType::kHALF) {

      CustomLinearKernel<<<bolck_num, blocksize, 0, stream>>>(
      (__half *)inputs[0], weight_half_gpu_, bias_half_gpu_,
      (__half *)outputs[0], blocksize, n_, len);

    } else {
      printf("Unsupport datatype!\n");
    }

  }


  return 0;
}

REGISTER_TENSORRT_PLUGIN(linearPluginCreator);