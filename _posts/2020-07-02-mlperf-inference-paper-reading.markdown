# MLPerf Inference Paper Reading

## 1. MLPerf Supporting Organizations

1. 28 Companies
    * Alibaba
    * AMD
    * ARM
    * Baidu
    * Google
    * Huawei
    * Intel
    * NVDIA
    * ...
2. 500+ discuusion group members
3. 7 Research Instituions
    * Harvard University
    * Stanford University
    * University of California, berkerly
    * ...

## 2. Deep Learning Benchmark Challeges

1. Diversity of models
    * Model name fail to uniquely describe a model
    * Community support, open source
2. Deployment-Senario Diversity
3. Inference-System Diversity
4. Something addtions
    * require industry-wide input
    * using latency other than MACs
    * accuracy is crucial

## 3. Contributions

1. Models Chosen
2. Identify Deep Learning realistic senarios
3. Defines Metrics
4. Defines allows and prohibited techniques, benchmarking rules
5. Extendable

## 4. MLPerf Inferenc Benchmark Design

1. **Models and datasets chosen**
    * lightweigh and heavyweight
    * Vision and Language
2. **Defines quanlity targets**
    * per-model quanlity targets
    * within 1% FP32 reference model's accuracy
    * no-retrainning
    * mobilenet within 2% FP32 reference accuary, 22.0 mAP
3. **Realistic End-User Scenarios**
    * Single-stream
    * Multistream
    * Server
    * Offline
4. **Statistically Confident Tail-Latency Bounds**
    * Sample size defined by tail-latency percentage, confidence and margin
    * run at least 60 seconds for dynamic voltage and freqency scaling(DVFS) consideration

## 5. Comparasion

* **AI Benchmark**
Only foncus on Android smartphones and only measure latency
* **EEMBC MLMark**
Only measure performance and accuracy of embedded inference devices, fixed best batch size
* **Fathom**
Focus on throughput rather than accuracy
* **AI Matrix**
Alibaba's Standard, focus on basic operation like convolution and matrix multiplication
* **DeepBench**
fail to addess the complexity of full models
* **TBD(Training Benchmarks for DNNs)**
Focus on ML training, focus on GPU only
* **DAWNBench**
Inspire MLPerf

## 6. [MLPerf Inference Summary](https://mlperf.org/inference-overview)
