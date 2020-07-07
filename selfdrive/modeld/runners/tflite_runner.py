#!/usr/bin/env python3
# TODO: why are the keras models saved with python 2?
from __future__ import print_function

#import tensorflow as tf  # pylint: disable=import-error
import os
import sys
import numpy as np
import tflite_runtime.interpreter as tflite

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

def run_loop(interpreter):
  #ishapes = [[1]+ii.get("shape")[1:] for ii in interpreter.get_input_details()]
  ishapes = [ii.get("shape") for ii in interpreter.get_input_details()]
  print("ready to run tflite model", ishapes, file=sys.stderr)
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  index = 0
  for shp in ishapes:
    ts = np.product(shp)
    #print("reshaping %s with offset %d" % (str(shp), offset), file=sys.stderr)
    interpreter.set_tensor(input_details[index]['index'], input_data)
    index += 1
  interpreter.invoke()
  ret = []
  for x in range(12):
    output = interpreter.get_tensor(interpreter.get_output_details()[x]['index']);
    print(x, " ", output)
    ret.append(output)
  #print(ret, file=sys.stderr)
  for r in ret:
    write(r)

if __name__ == "__main__":           
  #print(tf.__version__, file=sys.stderr)
  # limit gram alloc
  #gpus = tf.config.experimental.list_physical_devices('GPU')
  #if len(gpus) > 0:
  #  tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

  interpreter = tflite.Interpreter(model_path=sys.argv[1])
  interpreter.allocate_tensors()
  print(interpreter, file=sys.stderr)

  run_loop(interpreter)

