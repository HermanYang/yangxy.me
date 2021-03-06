# MLPerf Introduction

## 背景：
深度学习浪潮以降，对机器学习对算法需求越来越旺盛，新算法层出不穷，为此设计的各种优化方式如硬件优化，以及支持的软件库和各类深度学习框架如雨后春笋出现。寻找经典的代表算法，以体系结构中立的方式，可复现的来描述一个集合软硬件优化的推理系统将个巨大挑战。MLPerf Inference就是为解决推理系统性能比较而生，由30多个世界知名人工智能公司联合设计的一套规则，这套规则用于确保在不同体系结构上的推理性能的可比性。

1. 模型的选择，不同模型针对不同场景，有的对时延敏感，对能耗敏感，有的对精度敏感，因此不同模型的衡量意义不同，
2. 场景多样性
3. 推理系统多样性

1. 选择了经典的有代表性的算法，确保了算法的可用性和可复现性
模型算法使用名字难以唯一确定，比如ResNet-50，不足以描述一个确切的模型，因此各个厂商所谓ResNet-50的性能不具备可比性，MLPerf选定了一系列模型，这些模型是开源的，方便获取的，使用相同模型进行性能对比；
2.  总结了实际的应用场景
3.  为不同场景设计了不同性能描述指标
4.  定义了足够宽松的规则来展现软件和硬件的性能
5.  设计的性能描述方式可以允许模型快速演进，新模型可以很容易引入MLPerf而具有以上的性能描述

, lowering their tail latency
is important because: (1) completing them faster frees up
capacity for other jobs to run; (2) lower tail latency improves performance predictability and user satisfaction;
and (3) batch jobs may be rendered useless if they take
excessively long.

[MLPerf PPT](https://scholar.harvard.edu/files/mlperf_talk_for_sarc.pdf)