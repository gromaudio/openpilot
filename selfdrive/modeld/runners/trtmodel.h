#ifndef TRTMODEL_H
#define TRTMODEL_H

#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include "runmodel.h"
#include "NvInfer.h"
#include <cuda_runtime.h>
#include <cuda.h>

typedef nvinfer1::Dims3 Dims3; 

#define DIMS_0(x) x.d[0]
#define DIMS_1(x) x.d[1]
#define DIMS_2(x) x.d[2]
#define DIMS_3(x) x.d[3]


class TRTModel : public RunModel {
public:
  TRTModel(const char *path, float *output, size_t output_size, int runtime);
	~TRTModel();
  void addRecurrent(float *state, int state_size);
  void addDesire(float *state, int state_size);
  void addTrafficConvention(float *state, int state_size);
  void execute(float *net_input_buf, int buf_size);

  bool cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size );

protected:
  struct cudaLayer
  {
    Dims3 dims;
    uint32_t size;
    uint32_t binding;
    float* CPU;
    float* CUDA;
  };

  class Logger : public nvinfer1::ILogger     
  {
    void log( Severity severity, const char* msg ) override
    {
      if(severity != Severity::kINFO /*|| mEnableDebug*/)
        printf("%s\n", msg);
    }
  } gLogger;



private:

  float *output;
  size_t output_size;

  float *rnn_input_buf = NULL;
  int rnn_state_size;
  float *desire_input_buf = NULL;
  int desire_state_size;
  float *traffic_convention_input_buf = NULL;
  int traffic_convention_size;

  nvinfer1::IRuntime* runtime;
  nvinfer1::ICudaEngine* engine;
  nvinfer1::IExecutionContext* context;
  cudaStream_t stream;

  void** bindings;

  std::vector<cudaLayer> input_layers;
  cudaLayer output_layer;
};

#endif

