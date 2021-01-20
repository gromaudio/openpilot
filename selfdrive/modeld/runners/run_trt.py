import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda 
import datetime
import time
import os
import sys

input_size = 32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

if __name__ == "__main__":
    with open("supercombo_16.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
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

    print('--> Running model')
    for x in range(5):
        cuda.memcpy_htod(in_gpu_0, np.array(np.random.random_sample(engine.get_binding_shape(0)), dtype=np.float32))
        cuda.memcpy_htod(in_gpu_1, np.array(np.random.random_sample(engine.get_binding_shape(1)), dtype=np.float32))
        cuda.memcpy_htod(in_gpu_2, np.array(np.random.random_sample(engine.get_binding_shape(2)), dtype=np.float32))
        cuda.memcpy_htod(in_gpu_3, np.array(np.random.random_sample(engine.get_binding_shape(3)), dtype=np.float32))
        context.execute(1, [int(in_gpu_0), int(in_gpu_1), int(in_gpu_2), int(in_gpu_3), int(out_gpu)])
        cuda.memcpy_dtoh(out_cpu, out_gpu)
    print('--> Warmup done')

    iterations = 100
    total = 0
    for x in range(iterations):
        start_time = datetime.datetime.now()
        cuda.memcpy_htod(in_gpu_0, np.array(np.random.random_sample(engine.get_binding_shape(0)), dtype=np.float32))
        cuda.memcpy_htod(in_gpu_1, np.array(np.random.random_sample(engine.get_binding_shape(1)), dtype=np.float32))
        cuda.memcpy_htod(in_gpu_2, np.array(np.random.random_sample(engine.get_binding_shape(2)), dtype=np.float32))
        cuda.memcpy_htod(in_gpu_3, np.array(np.random.random_sample(engine.get_binding_shape(3)), dtype=np.float32))
        context.execute(1, [int(in_gpu_0), int(in_gpu_1), int(in_gpu_2), int(in_gpu_3), int(out_gpu)])
        cuda.memcpy_dtoh(out_cpu, out_gpu)
        elapsed = datetime.datetime.now() - start_time
        total += elapsed.total_seconds() * 1000
        print("onnx execution time", int(elapsed.total_seconds()*1000), file=sys.stderr)
	    #show_outputs(outputs)
    print("bench: ", total / iterations, 1000 / (total / iterations))
    print('done')
    


# tensorrt docker image: docker pull nvcr.io/nvidia/tensorrt:19.09-py3 (See: https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt/tags)
# NOTE: cuda driver >= 418