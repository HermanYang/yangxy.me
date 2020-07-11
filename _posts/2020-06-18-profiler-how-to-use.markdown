# Tensorflow Models Profiler

## 1. 背景

目前推理和训练均有性能统计的需求，性能统计会占用大量人力时间，如果将性能统计功能加入各个模型当中，就能达到快速生成性能统计数据的目的。
为了可以快速和正确地采集性能数据，需要有标准的性能数据采集步骤，集成了这些步骤后，模型需要可方便地关闭性能采集以免影响性能，同时性能采集的代码要尽可能简洁。
为此，封装了性能采集接口，帮助工程师编写模型性能采集功能。

## 2. 性能数据分层

* 应用层：采集模型的数据准备时间，数据后处理时间，程序整体时间等应用层的数据；
* 框架层：采集模型算子的性能数据，包括计算时间，消耗内存等；
* 运行时层：采集CNRT，CNML，CNNL以及驱动层等运行时的时间消耗；
* 硬件层：采集硬件计算时间，硬件利用率，硬件IO状况数据；

**目前，可以采集到的有应用层，框架层以及部分运行时层的性能数据，硬件层的数据目前仍然无法采集到，需要魔改TF来集成系统工具组的接口。**

## 3. 使用方法

1.初始化
Prifler是一个全局单例，在使用前初始化即可；

``` python
  from cambricon_example.utils.profiler import Profiler
  profiler = Profiler()
```

运行模型前通过控制环境变量打开或者关闭Profile功能

``` bash
export TENSORFLOW_MODELS_TRACE_LEVEL=2 # 打开Profile功能
export TENSORFLOW_MODELS_TRACE_LEVEL=0 # 关闭Profile功能
```

2.采集应用层数据

``` python
  profiler.activity_start(activity_name, step, meta)
  ...
  doSomeActivity()
  ...
  profiler.activity_end(activity_name, step, meta)
```

3.采集框架层数据
目前只考虑了两种模型运行方式，一种是Keras运行方式，另一种是普通的Session Run方法。

``` python
# Keras
  ...
  options = tf.compat.v1.RunOptions(trace_level = profiler.get_tf_trace_level())
  model.compile(..., options = options, run_metadata = profiler.get_tf_runmetadata(graph_name))
  model.fit(..., callbacks = [profiler.get_collect_runmetadata_callback(graph_name)])
  ...
  profiler.finalize()
```

``` python
# Session Run
  ...
  options = tf.compat.v1.RunOptions(trace_level = profiler.get_tf_trace_level())
  profiler.step_start(graph_name, step)
  session.Run(..., options = options, run_metadata=profiler.get_tf_runmetadata(graph_name))
  profiler.step_end(graph_name, step)
  ...
```

4.生成性能报告

``` python
  profiler.finalize()
```

## 4. 性能报告

1. 生成的目录

生成Timeline数据与timeline文件夹，可以查看每个step的timeline数据，数据总结在summary.md文件和summary.json文件里面。

``` text
profile/
  timeline/
    model_step_0
    model_step_1
    ...
  step_stats/
    model_step_0
    model_step_1
    ...
  summary.json
  summary.md
 ```

2. summary.json内容

``` json
 {
    "app_stats": {
      "[step]":{
        "[activity_name]":{
          "duration":100,
          "meta": "This is Vgg session run"
        }
      }
    },
    "op_stats":{
      "[model]":{
        "[step]":{
          "[ip]":{
            "[device]":{
              "[node name]":{
                "duration":0,
                "output":{
                  "[slot]":{
                    "shape":[1, 2, 4],
                    "type":"Int32"
                  }
                },
                "total_bytes":128
              }
            }
          }
        }
      }
    }
}
```

3. vgg16 summary.json 例子

``` json
 {
    "app_stats": {
      "0":{
        "Vgg Session Run":{
          "duration":100,
          "meta": "This is Vgg session run"
        }
      }
    },
    "op_stats":{
      "vgg16":{
        "0":{
          "localhost":{
            "CPU":{
              "op_name":{
                "duration":0,
                "output":{
                  "0":{
                    "shape":[1, 2, 4],
                    "type":"Int32"
                  }
                },
                "total_bytes":128
              }
            }
          }
        }
      }
    }
  }
  ```