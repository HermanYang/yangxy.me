# NVDIA Profiler Introduction

## Overview
1. [Visual Profiler, aks nvvp](https://docs.nvidia.com/cuda/profiler-users-guide/#nvprof-overview)
2. [nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/#nvprof-overview)
The easiest profiler to use is nvprof, a command-line light-weight profiler which presents an overview of the GPU kernels and memory copies in your application. You can use nvprof as below:```nvprof python run_inference.py```
3. Nsight Systems/Nsight Compute
[Nsight VS Visual Profiler and nvprof](https://docs.nvidia.com/cuda/profiler-users-guide/#migrating-to-nsight-tools)
4. [NVIDIA Tools Extension, aks nvtx](https://docs.nvidia.com/cuda/profiler-users-guide/#nvtx)
TensorFlow inside the NVIDIA container is built with NVTX ranges for TensorFlow operators. This means every operator (including TRTEngineOp) executed by TensorFlow will appear as a range on the visual profiler which can be linked against the CUDA kernels executed by that operator. This way, you can check the kernels executed by TensorRT, the timing of each, and compare that information with the profile of the original TensorFlow graph before conversion.
5. [DLProf](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html)
**DLProf is a wrapper tool around Nsight Systems** that correlates profile timing data and kernel information to a Machine Learning model. The correlated data is presented to a Data Scientist in a format that can be easily digested and understood by the Data Scientist. The results highlight GPU utilization of model and DL/ML operations. The tools provide different reports to aid in identifying bottlenecks and Tensor Core usage.
6. Tensorflow Profiler

## Some Articles
[TensorRT Profilers Introduction](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/#profiling-tools)
Introduces NVIDIA profiler tools to profile TensorRT programs.

[High Performance inference with TensorRT Integration](https://medium.com/tensorflow/high-performance-inference-with-tensorrt-integration-c4d78795fbfe)
1. tensorrt conversion
2. quantization and calibiration
3. use nvprof to measure performance

## Personal Throught
1. Nsight System ~ nvprof & nvvp(Virtual Profiler)
2. DLProf = Nsight + NVTX + Tensorboard Plugin
3. DLProf Disadvantage:
    1. Missing Application Layer profile data
    2. Missing Framework Layer Profile data

## Profiler Components

```puml
@startuml
[Profiler] as profiler
[Python Model] as model
[Tensorflow Profiler] as tf_profiler
[CNPAPI] as cnpapi

model --> profiler
profiler --> tf_profiler
profiler --> cnpapi

@enduml
```

## Profiler Sequences

```puml
@startuml

Model --> Profiler: Preprocess
Model --> "Tensorflow/OpenCV/etc.": Preprocess
Model --> Profiler: Preprocess End
Profiler --> Profiler: Record
Model --> Profiler: Session Run
Profiler --> "Tensorflow Profiler": Collect RunMetaData
"Tensorflow Profiler" --> Profiler: Op Stats
Profiler --> "CNPAPI": Collect Runtime Stats
"CNPAPI" --> Profiler: Runtime Stats
Profiler --> "CNPAPI": Collect Hardward Stats
"CNPAPI" --> Profiler: Hardward Stats
Model --> Profiler: Step End
Model --> Profiler: Postprocess
Model --> "Tensorflow/OpenCV/etc.": Postprocess
Profiler --> Profiler: Record
Model --> Profiler: Postprocess End
Model --> Profiler: Finalize
Profiler --> Profiler: Generate Stats
Profiler --> Profiler: Generate Timelines
Profiler --> Profiler: Generate Summary

@enduml
```
