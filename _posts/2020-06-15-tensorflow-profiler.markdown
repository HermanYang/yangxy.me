## Class Hierachy
**Event Collector**
```puml
enum EventCategory {
  ShceduleClosure
  RunClosure
  Compute
}

class EventCollector {
  + isEnabled()
  + RecordEvent()
  + SetCurrentThreadName()
  + StartRegin()
  + StopRegion()
}

class ScopeRegion {
  + ScopeRegion()
  + ~ScopeRegion()
}

ScopeRegion o-- EventCollector
```

**GPU Device Tracerr**
```puml
class DeviceInfo {
  int ordinal
  string name
  int num_contexts
}

class ContextInfo {
  int index
  DeviceInfo dev_info
  int num_streams
  CUevent end_event
}

class StreamInfo {
  string name
  int index
  Contextinfo ctx_info
}

class CudaEventCollector {
  + AddStreamInfo(context, stream, name)
  + Collect()
  + CudaEventCollector
  + GetContextInfo
  + GetElapsedTimeUs
  + GetMemcpyName
  + InitializedDeviceInfos()
  + SaveRecord(record)
  + SaveStats()
  + Synchronize()
}

class CudaEventRecorder {
    ConsumeKernelRecords()
    ConsumeMemcpyRecords()
    StartKernel()
    StartMemcpy()
    StopKernel()
    StopMemcpy()
}

class CuptiCallbackHook {
  CuptiCallback()
  CuptiCallbackHook()
  DriverApiEnterCallback
  DriverApiExitCallback
  Enable
  GetMemoryType
  StartMemcpy
  StartMemcpyAsync
}

class StepStatsCollector
class ProfilerInterface
class DeviceTracer {
  + Start()
  + Stop()
  + CollectData(RunMetadata)
}

ProfilerInterface <|-- DeviceTracer 

StreamInfo o-- ContextInfo
ContextInfo o-- DeviceInfo
CudaEventCollector o-- StreamInfo
CudaEventCollector o-- CudaEventRecorder 
CudaEventCollector o-- StepStatsCollector
DeviceTracer o-- CuptiCallbackHook
DeviceTracer o-- CudaEventCollector
```

**MLU Device Tracer**
```puml
class MallocDetails {
  uint64 num_bytes
}

class MemcpyDetails {
    size num_bytes
    Dir direction
    bool async
}

class MemcpyPeerDetails {
  size num_bytes
  int src_device
  int dst_device
}

enum CnpapiTracerEventType {
  Unsupported
  kernel
  memcpyPeer
  MemoryAlloc
  Generic
}

enum CnpapiTracerEventSource {
  Callback
  Activity
  Notifier
}

class KernelDetails{
  string kernel_name
}

class CnpapiTracerEvent {
  CnpapiTracerEventType type
  CnpapiTracerEventSource source
  string name
  uint64 start_time_ns
  uint64 end_time_ns
  uint64 schedule_time_ns
  uint32 device_id
  uint32 thread_id
  string anotation
  KernelDetails kernel_details
  MemcpyDetails memcpy_details
  MemcpyPeerDetail memcpy_peer_details
  MallocDetails malloc_details
}

class CnpapiTracerOptions {
  bool enable_activity_api
  bool enable_event_based_activity
  bool required_callback_api_events
  vector cnml_cbids_selected
  vector cnrt_cbids_selected
  vector activities_selected
  bool cnpapi_finalize
}

class CnpapiInterface {

}

class CnpapiLoader {

}

class CnpapiWrapper {
  CnpapiLoader cnpapi_loder
}

class CnpapiTracer {
  CnpapiSopaApiHook cnpapi_sopa_api_hook_
  cnpapi_SubscriberHandler subscripber_
  CnpapiInterface cnpapi_interface_
  CnpapiTracerCollector collector_
}

interface ProfilerInterface {
  +Start()
  +Stop()
  +CollectData(RunMetadata* )
}

class MLUTracer {
  CnpapiTracer cnpapi_tracer
  CnpapiTracerOptions options
  CnpapiTracerCollectorImpl cnpapi_collector
}

interface CnpapiTracerCollector {
  +void AddEvent(CnpapiTracerEvent&&)
  +void onEventsDropped(string reason, uint32 num_events)
  +void Flush()
  +AnootationMap* annotation_map()
  +uint32 GetDeviceId(Dev dev)
}

class CnpapiTracerCollectorImpl {

}

class CnpapiSopaApiHookWithDeviceEvent {
  collector_
  options_
  sopa_vent_recorder_
  Fulsh()
  OnSopaApiEnter()
  OnSopaApiExit()
}

class CnpapiSopaApiHookWithHostEvent {
  collector_
  options_
  sopa_vent_recorder_
  Fulsh()
  OnSopaApiEnter()
  OnSopaApiExit()
}

class CssnpapiApiTracingDisabler {

}

class SopaEventRecorder{
  CreateAndPlaceNotifier(MLUCnrtNotifier **, MLUCnrtQueue *)
  Flush()
  RecordEvent(const EventRecord &)
  SaveRecord(const EventRecord &)
}

class EventRecord {
  CnpapiTracerEventType type
  string name
  uint64 start_timestamp
  MLUCnrtQueue* queue
  MLUCnrtNotifier* start_notifier
  MLUCnrtNotifier* end_notifier
  uint32 device_id
  uint32 thread_id
  string anotation
  KernelDetails kernel_details
  MemcpyDetails memcpy_details
}

class QueueInfo {
   uint64 end_walltime_us
   bool synchronized 
   MLUCnrtNotifier end_notifier
}

class AnnotationStack {
  +static PushAnnotation(name)
  +static PopAnnotation(name)
  static* 
}

CnpapiTracer o-- CnpapiTracerCollector
CnpapiTracer o-- CnpapiInterface
CnpapiTracerEvent o-- CnpapiTracerEventType
CnpapiTracerEvent o-- CnpapiTracerEventSource
CnpapiTracerEvent o-- KernelDetails
CnpapiTracerEvent o-- MemcpyDetails
CnpapiTracerEvent o-- MemcpyPeerDetails
CnpapiTracerEvent o-- MallocDetails
CnpapiInterface <|-- CnpapiWrapper 
CnpapiWrapper o-- CnpapiLoader
MLUTracer o-- CnpapiTracer
MLUTracer o-- CnpapiTracerOptions
MLUTracer o-- CnpapiTracerCollectorImpl
ProfilerInterface <|-- MLUTracer 
CnpapiTracerCollector <|-- CnpapiTracerCollectorImpl
```

## Tensorflow Profiing Sequences
### Local Session
**1. Overall**
```puml
@startuml
"DirectSession::RunInternal" -> ProfilerSession: Create
ProfilerSession -> HostTracer: Create
ProfilerSession -> DeviceTracer: Create
ProfilerSession -> HostTracer: Start
ProfilerSession -> DeviceTracer: Start
"DirectSession::RunInternal" -> StepStatsCollector: new
"DirectSession::RunInternal" -> ProfilerSession: CollectData
loop 
"DirectSession::RunInternal" -> ExecutorState: Process
"ExecutorState" -> NodeExecStats: SetExecutorStarted
"ExecutorState" -> NodeExecStats: SetScheduled
"ExecutorState" -> NodeExecStats: SetComputeStarted
"ExecutorState" -> Device: Compute
"ExecutorState" -> NodeExecStats: SetComputeEnded
"ExecutorState" -> NodeExecStats: SetMemory
"ExecutorState" -> NodeExecStats: SetexecutorEnded
end
"DirectSession::RunInternal" -> StepStatsCollector: Finalize
StepStatsCollector -> NodeExecStats: Finalize
"DirectSession::RunInternal" -> ProfilerSession: Destruction
ProfilerSession -> Profiler: Stop
@enduml
```
**2. Memory Profiling**

```puml
enum AllocRecord {
  int alloc_bytes
  int alloc_micros
}
```

```puml
ExecutorState -> Device: Compute
Device -> OpKernel: Compute
OpKernel -> TrackingAllocator: Create
Device -> TrackingAllocator: AllocateRaw
TrackingAllocator -> TrackingAllocator: AddAllocateRecord
Device -> TrackingAllocator: DeallocateRaw
TrackingAllocator -> TrackingAllocator: AddAllocateRecord
ExecutorState -> NodeExecStats: SetMemory
NodeExecStats -> OpKernel: ConsumeWrappedAllocators
```

**2. Activity/ScropedRegion/Annotation**
For Compute, Run Closure events
```puml
ScopeRegion -> EventCollector:StartRegion
ScopeRegion -> EventCollector:StopRegion
```

Any General Code sn
```puml
TraceMe -> TraceMeRecorder:Record
```

```puml
ScopedAnnotation -> Annotation:PushAnnotation
ScopedAnnotation -> Annotation:PopAnnotation
Annotation -> "Thread Local":ThreadAnnotation
XXX -> Annotation:CurrentAnnotation
```
**3. GPU Profiler**

LaunchKernel,
Memcpy
MemcpyAsync
MemcpyHtoD
MemcpyHtoDAsync
MemcpyDtoH
MemcpyDtoHAsync
MemcpyDtoD
MemcpyDtoDAsync

```puml
ProfileSession -> DeviceTracer: Start
DeviceTracer -> ScoptedAnnotation: Enbale
DeviceTracer -> CuptiCallbackHook: new
CuptiCallbackHook -> CUPTI: cputiSubscribe
CuptiCallbackHook -> CUPTI: cputiEnbaleCallback
CUPTI -> CuptiCallbackHook: CuptiCallback
CuptiCallbackHook -> CuptiCallbackHook: DriverApiEnterCallback
CuptiCallbackHook -> CudaEventRecorder: StartKernel/StartMemcpy/StopKernel/StopMemcpy
CudaEventRecorder -> Annotation: CurrentAnnocation
ProfileSession -> DeviceTracer: CollectData
DeviceTracer -> CudaEventCollector: Collect
CudaEventCollector -> CudaEventRecorder: ConsumeKernelRecords
CudaEventCollector -> CudaEventRecorder: ConsumeMemcpyRecords
CudaEventCollector -> StepStatsCollector: Save
```
**4. MLU Profiler**
cnmlCompileBaseOp
cnmlCompileFusionOp
cnrtInvokeKernel
cnrtInvokeRuntimeContext
cnrtDestroyQueue
cnrtSyncQueue
cnrtMemcpy
cnrtMemcpyPeer
cnrtMalloc
cnrtMemcpyAsync

```puml
ProfileSession -> MLUTracer: Create
MLUTracer -> CnpapiInterface: Create
ProfileSession -> MLUTracer: Start
MLUTracer -> CnpapiTracerCollector: new
MLUTracer -> AnnotationStack: Enable
MLUTracer -> CnpapiTracer: Enable
CnpapiTracer -> CnpapiCnApiHookWithDeviceEvent:new
CnpapiTracer -> CnpapiCnApiHookWithHostEvent:new
CnpapiTracer -> CnpapiInterface: Subscribe
CnpapiInterface -> Papi: subscribe
Papi -> CnpapiTracer: HandleCallback
CnpapiTracer -> CnEventRecorder: StarttKernelEvent/StopEvent
CnEventRecorder -> CnEventRecorder: Record Event
ProfileSession -> MLUTracer: CollectData
MLUTracer -> CnpapiTracerCollector: CollectData


ProfileSession -> MLUTracer: Stop
MLUTracer -> AnnotationStack: Disable
MLUTracer -> CnpapiTracer: Disable

```


### Remote Session