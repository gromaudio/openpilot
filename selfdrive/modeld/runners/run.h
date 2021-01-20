#ifndef RUN_H
#define RUN_H

#include "runmodel.h"
#include "snpemodel.h"

#ifdef QCOM
  #define DefaultRunModel SNPEModel
#else
  #ifdef USE_ONNX_MODEL
    #include "onnxmodel.h"
    #define DefaultRunModel ONNXModel
  #else
    #include "trtmodel.h"
    #define DefaultRunModel TRTModel
  #endif
#endif

#endif
