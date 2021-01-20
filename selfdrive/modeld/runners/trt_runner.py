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

