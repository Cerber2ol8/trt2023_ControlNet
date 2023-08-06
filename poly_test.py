from collections import OrderedDict

import numpy as np
import polygraphy.backend.trt as p
import tensorrt as trt

onnxFile = "./onnxsim_model/FrozenCLIPEmbedder.onnx"
trtFile = "./trt_dir/FrozenCLIPEmbedder.plan"
cacheFile = "./tmp/FrozenCLIPEmbedder.cache"

builder, network, parser = p.network_from_onnx_path(onnxFile)

input_name = 'input_ids'
input_size = [1 ,77]

profileList = [p.Profile().add(input_name, input_size, input_size, input_size)]

builderConfig = p.CreateConfig( \
    tf32=False,
    fp16=True,
    int8=False,
    profiles=profileList,
    calibrator=None,
    precision_constraints=None,
    strict_types=False,
    load_timing_cache=None,
    algorithm_selector=None,
    sparse_weights=False,
    tactic_sources=None,
    restricted=False,
    use_dla=False,
    allow_gpu_fallback=False,
    profiling_verbosity=None,
    memory_pool_limits={trt.MemoryPoolType.WORKSPACE:1<<30},
    refittable=False,
    preview_features=None,
    engine_capability=None,
    direct_io=False,
    builder_optimization_level=None,
    fp8=False,
    hardware_compatibility_level=None,
    max_aux_streams=4,
    version_compatible=False,
    exclude_lean_runtime=False)

engineString = p.engine_from_network([builder, network], config=builderConfig, save_timing_cache=cacheFile)

p.save_engine(engineString, path=trtFile)

runner = p.TrtRunner(engineString, name=None, optimization_profile=0)

runner.activate()

output = runner.infer(OrderedDict([(input_name, np.ascontiguousarray(np.random.rand(1, 77).astype(np.int32)))]), check_inputs=True)

runner.deactivate()

print(output)
"""
methods of polygraphy.backend.trt:
'Algorithm'
'BytesFromEngine'
'Calibrator'
'CreateConfig'
'CreateNetwork'
'EngineBytesFromNetwork'
'EngineFromBytes'
'EngineFromNetwork'
'LoadPlugins'
'ModifyNetworkOutputs'
'NetworkFromOnnxBytes'
'NetworkFromOnnxPath'
'OnnxLikeFromNetwork'
'Profile'
'SaveEngine'
'ShapeTuple'
'TacticRecorder'
'TacticReplayData'
'TacticReplayer'
'TrtRunner'
'__builtins__'
'__cached__'
'__doc__'
'__file__'
'__loader__'
'__name__'
'__package__'
'__path__'
'__spec__'
'algorithm_selector'
'bytes_from_engine'
'calibrator'
'create_config'
'create_network'
'engine_bytes_from_network'
'engine_from_bytes'
'engine_from_network'
'get_trt_logger'
'load_plugins'
'loader'
'modify_network_outputs'
'network_from_onnx_bytes'
'network_from_onnx_path'
'onnx_like_from_network'
'profile'
'register_logger_callback'
'runner'
'save_engine'
'util'

"""