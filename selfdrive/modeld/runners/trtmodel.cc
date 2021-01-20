#include "trtmodel.h"

#include <stdio.h>
#include <string>
#include <string.h>
#include <poll.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cassert>

#include "common/util.h"
#include "common/utilpp.h"
#include "common/swaglog.h"

/*
#!/usr/bin/env python3.6
# TODO: why are the keras models saved with python 2?
from __future__ import print_function

import os
import sys
import numpy as np
import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda 
import datetime
import time
import os
import sys
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def read(sz):
  dd = []
  gt = 0
  while gt < sz * 4:
    st = os.read(0, sz * 4 - gt)
    assert(len(st) > 0)
    dd.append(st)
    gt += len(st)
  return np.frombuffer(b''.join(dd), dtype=np.float32)

def write(d):
  os.write(1, d.tobytes())

def run_loop(model):
  with open(model, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
  context = engine.create_execution_context()

  in_size_0 = trt.volume(engine.get_binding_shape(0))
  in_size_1 = trt.volume(engine.get_binding_shape(1))
  in_size_2 = trt.volume(engine.get_binding_shape(2))
  in_size_3 = trt.volume(engine.get_binding_shape(3))

  out_size = trt.volume(engine.get_binding_shape(4))

  in_dtype = trt.nptype(engine.get_binding_dtype(0))
  out_dtype = trt.nptype(engine.get_binding_dtype(4))

  in_cpu_0 = cuda.pagelocked_empty(in_size_0, in_dtype)
  in_cpu_1 = cuda.pagelocked_empty(in_size_0, in_dtype)
  in_cpu_2 = cuda.pagelocked_empty(in_size_0, in_dtype)
  in_cpu_3 = cuda.pagelocked_empty(in_size_0, in_dtype)

  out_cpu = cuda.pagelocked_empty(out_size, out_dtype)
  # allocate gpu mem
  in_gpu_0 = cuda.mem_alloc(in_cpu_0.nbytes)
  in_gpu_1 = cuda.mem_alloc(in_cpu_1.nbytes)
  in_gpu_2 = cuda.mem_alloc(in_cpu_2.nbytes)
  in_gpu_3 = cuda.mem_alloc(in_cpu_3.nbytes)
  out_gpu = cuda.mem_alloc(out_cpu.nbytes)
  stream = cuda.Stream()

  print('--> Running model', file=sys.stderr)
  while 1:
    start_time = datetime.datetime.now()
    #cuda.memcpy_htod(in_gpu_0, np.array(np.random.random_sample(engine.get_binding_shape(0)), dtype=np.float32))
    #cuda.memcpy_htod(in_gpu_1, np.array(np.random.random_sample(engine.get_binding_shape(1)), dtype=np.float32))
    #cuda.memcpy_htod(in_gpu_2, np.array(np.random.random_sample(engine.get_binding_shape(2)), dtype=np.float32))
    #cuda.memcpy_htod(in_gpu_3, np.array(np.random.random_sample(engine.get_binding_shape(3)), dtype=np.float32))

    cuda.memcpy_htod(in_gpu_0, read(np.product(engine.get_binding_shape(0))).reshape(engine.get_binding_shape(0)))
    cuda.memcpy_htod(in_gpu_1, read(np.product(engine.get_binding_shape(1))).reshape(engine.get_binding_shape(1)))
    cuda.memcpy_htod(in_gpu_2, read(np.product(engine.get_binding_shape(2))).reshape(engine.get_binding_shape(2)))
    cuda.memcpy_htod(in_gpu_3, read(np.product(engine.get_binding_shape(3))).reshape(engine.get_binding_shape(3)))

    context.execute(1, [int(in_gpu_0), int(in_gpu_1), int(in_gpu_2), int(in_gpu_3), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    elapsed = datetime.datetime.now() - start_time
    for r in out_cpu:
      write(r)    
    print("trt execution time", int(elapsed.total_seconds()*1000), file=sys.stderr)

if __name__ == "__main__":
  run_loop(sys.argv[1])


*/

TRTModel::TRTModel(const char * path, float * _output, size_t _output_size, int todoChange) {
  output = _output;
  output_size = _output_size;

  char tmp[1024];
  strncpy(tmp, path, sizeof(tmp));
  strstr(tmp, ".dlc")[0] = '\0';
  strcat(tmp, ".trt");
  LOGW("loading model %s", tmp);


  std::ifstream engineFile("/home/nano/openpilot/models/supercombo.trt", std::ios::binary);
  if (!engineFile) {
    LOGE("Error opening engine file %s", tmp);
    //err << "Error opening engine file: " << engine << std::endl;
    return;
  }

  engineFile.seekg(0, engineFile.end);
  long int fsize = engineFile.tellg();
  engineFile.seekg(0, engineFile.beg);

  std::vector < char > engineData(fsize);
  engineFile.read(engineData.data(), fsize);
  if (!engineFile) {
    LOGE("Error opening engine file %s", tmp);
    //return nullptr;
  }

  runtime = nvinfer1::createInferRuntime(gLogger);
  engine = runtime -> deserializeCudaEngine(engineData.data(), fsize, nullptr);
  context = engine -> createExecutionContext();

  printf("CUDA engine context initialized with %u bindings\n", engine -> getNbBindings());

  for (int i = 0; i < 5; i++) {
    const nvinfer1::Dims inputDims = engine -> getBindingDimensions(i);
    size_t inputSize = sizeof(float) * DIMS_0(inputDims) * DIMS_1(inputDims);
    if (inputDims.nbDims >= 3)
      inputSize *= DIMS_2(inputDims);
    if (inputDims.nbDims >= 4)
      inputSize *= DIMS_3(inputDims);

    printf("layer %d: dims (n=%u c=%u h=%u w=%u) size=%zu\n", i, DIMS_0(inputDims), DIMS_1(inputDims), DIMS_2(inputDims),
      DIMS_3(inputDims), inputSize);
    void * memCPU = NULL;
    void * memCUDA = NULL;
    if (!cudaAllocMapped((void ** ) & memCPU, (void ** ) & memCUDA, inputSize)) {
      printf("failed to alloc CUDA mapped memory for tensorNet input, %zu bytes\n", inputSize);
      //return false;
    }

    cudaLayer layer;
    layer.CPU = (float * ) memCPU;
    layer.CUDA = (float * ) memCUDA;
    layer.size = inputSize;

    DIMS_0(layer.dims) = DIMS_0(inputDims);
    DIMS_1(layer.dims) = DIMS_1(inputDims);
    DIMS_2(layer.dims) = DIMS_2(inputDims);
    DIMS_3(layer.dims) = DIMS_3(inputDims);

    layer.binding = i;

    if (i != 4)
      input_layers.push_back(layer);
    else
      output_layer = layer;
  }

  const int nubindings = engine -> getNbBindings();
  const int bindingSize = nubindings * sizeof(void * );
  bindings = (void ** ) malloc(bindingSize);
  if (!bindings) {
    LOGE("failed to allocate %u bytes for bindings list\n", bindingSize);
    return;
  }

  memset(bindings, 0, bindingSize);

  for (uint32_t n = 0; n < input_layers.size(); n++)
    bindings[input_layers[n].binding] = input_layers[n].CUDA;

  bindings[output_layer.binding] = output_layer.CUDA;

  uint32_t flags = cudaStreamDefault;
  stream = NULL;
  if (cudaStreamCreateWithFlags( & stream, flags) != cudaSuccess)
    return;
}

TRTModel::~TRTModel() {
  // todo: close everything
  if (engine != NULL) {
    engine -> destroy();
    engine = NULL;
  }

  if (runtime != NULL) {
    runtime -> destroy();
    runtime = NULL;
  }
}

void TRTModel::addRecurrent(float * state, int state_size) {
  rnn_input_buf = state;
  rnn_state_size = state_size;
}

void TRTModel::addDesire(float * state, int state_size) {
  desire_input_buf = state;
  desire_state_size = state_size;
}

void TRTModel::addTrafficConvention(float * state, int state_size) {
  traffic_convention_input_buf = state;
  traffic_convention_size = state_size;
}

void TRTModel::execute(float * net_input_buf, int buf_size) {
  if (cudaMemcpy(input_layers[0].CUDA, net_input_buf, buf_size, cudaMemcpyHostToDevice) != cudaSuccess) {
    LOGE("failed to copy");
    return;
  }

  if (desire_input_buf != NULL) {
    if (cudaMemcpy(input_layers[1].CUDA, desire_input_buf, desire_state_size, cudaMemcpyHostToDevice) != cudaSuccess) {
      LOGE("failed to copy");
      return;
    }
  }
  if (traffic_convention_input_buf != NULL) {
    if (cudaMemcpy(input_layers[2].CUDA, traffic_convention_input_buf, traffic_convention_size, cudaMemcpyHostToDevice) != cudaSuccess) {
      LOGE("failed to copy");
      return;
    }
  }
  if (rnn_input_buf != NULL) {
    if (cudaMemcpy(input_layers[3].CUDA, rnn_input_buf, rnn_state_size, cudaMemcpyHostToDevice) != cudaSuccess) {
      LOGE("failed to copy");
      return;
    }
  }

  if (!context -> execute(1, bindings)) {
    LOGE("failed to execute TensorRT context on device\n");
    return;
  }


  if (cudaMemcpy(output_layer.CUDA, output_layer.CPU, output_layer.size, cudaMemcpyDeviceToHost ) != cudaSuccess) {
      LOGE("failed to copy");
      return;
  }

   memcpy (output, output_layer.CPU, output_layer.size);
}

bool TRTModel::cudaAllocMapped(void ** cpuPtr, void ** gpuPtr, size_t size) {
  if (!cpuPtr || !gpuPtr || size == 0)
    return false;

  //CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

  if (cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped) != cudaSuccess)
    return false;

  if (cudaHostGetDevicePointer(gpuPtr, * cpuPtr, 0) != cudaSuccess)
    return false;

  memset( * cpuPtr, 0, size);
  printf("cudaAllocMapped %zu bytes, CPU %p GPU %p\n", size, * cpuPtr, * gpuPtr);
  return true;
}