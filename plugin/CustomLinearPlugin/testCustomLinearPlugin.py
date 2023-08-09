import os
import ctypes
import numpy as np
from time import time_ns
import tensorrt as trt
import torch.nn as nn
import torch
from cuda import cudart



useFile         = False

soFilePath      = './CustomLinear.so'
nBS             = 1
nSL             = 77
nEmbedding      = 1280
nTime           = 100
epsilon         = 1e-6

np.random.seed(97)

npToTRT = {np.int8:trt.int8,np.float16:trt.float16,np.int32:trt.int32,np.float32:trt.float32}
npToPFT = {np.int8:trt.PluginFieldType.INT8,np.float16:trt.PluginFieldType.FLOAT16,
            np.int32:trt.PluginFieldType.INT32,np.float32:trt.PluginFieldType.FLOAT32}


weight = np.random.rand(nEmbedding).astype(np.float32)
bias = np.random.rand(nEmbedding).astype(np.float32)

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )
    
class CustomLinear(nn.Module):

    def __init__(self, weight, bias, eps=1e-5):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.from_numpy(weight))
        self.bias = nn.Parameter(torch.from_numpy(bias))

    @torch.inference_mode()
    def forward(self, x):
        r = self.weight * x + self.bias
        return r.numpy()



def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'CustomLinear':
            parameterList = []
            parameterList.append(trt.PluginField("weight", weight, trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run():
    testCase = "test<fp%s,bs=%d,sl=%d,nEmbed=%d>"%(['32','16'][0],nBS,nSL,nEmbedding)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.max_workspace_size = 22 << 30
    config.flags    = [0,1<<int(trt.BuilderFlag.FP16)][0]

    inputTensorList = []
    inputTensorList.append( network.add_input('input', trt.float32, [-1,77,nEmbedding]) )
    # inputTensorList.append( network.add_input('inputB', trt.float32, [256]) )
    # inputTensorList.append( network.add_input('inputA', trt.float32, [256]) )

    profile = builder.create_optimization_profile()
    profile.set_shape('input',[1,77,nEmbedding],[1,77,nEmbedding],[1,77,nEmbedding])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())
    pluginLayer.get_output(0).dtype = [trt.float32,trt.float16][0]

    network.mark_output(pluginLayer.get_output(0))
    
    engine = builder.build_engine(network, config)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[nBS,nSL,nEmbedding])
    # context.set_binding_shape(1,[nEmbedding])
    # context.set_binding_shape(2,[nEmbedding])

    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))



    # for i in range(network.num_layers):
    #     layer = network.get_layer(i)
    #     print(i, "%s,in=%d,out=%d,%s" % (str(layer.type)[10:], layer.num_inputs, layer.num_outputs, layer.name))
    #     for j in range(layer.num_inputs):
    #         tensor = layer.get_input(j)
    #         if tensor == None:
    #             print("\tInput  %2d:" % j, "None")
    #         else:
    #             print("\tInput  %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
    #     for j in range(layer.num_outputs):
    #         tensor = layer.get_output(j)
    #         if tensor == None:
    #             print("\tOutput %2d:" % j, "None")
    #         else:
    #             print("\tOutput %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
    # #exit()


    lTensorName = [engine.get_tensor_name(i) for i in range(engine.num_bindings)]

    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    bufferH.append( np.random.rand(nBS,nSL,nEmbedding).astype(np.float32).reshape(nBS,nSL,nEmbedding) * 2 - 1)
    # bufferH.append( np.ones(nEmbedding).astype(np.float32) )
    # bufferH.append( np.zeros(nEmbedding).astype(np.float32) )
    bufferH.append( np.empty(context.get_binding_shape(1),dtype=trt.nptype(engine.get_binding_dtype(1))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        



    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        context.set_tensor_address(lTensorName[i], bufferD[i])
    
    for i in range(nOutput):
        context.set_tensor_address(lTensorName[nInput+i], bufferD[nInput+i])

    context.execute_async_v3(0)


    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput+i].ctypes.data, bufferD[nInput+i], bufferH[nInput+i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nInput):
        temp = bufferH[i]
        print("inputH%d"%i, temp.shape,np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))))
        print(temp.reshape(-1)[:10])
        #print(temp)
    
    for i in range(nOutput):
        temp = bufferH[nInput+i]
        print("outputH%d"%i, temp.shape,np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))))
        print(temp.reshape(-1)[:10])
        #print(temp)
    

    for i in range(10):
        context.execute_async_v3(0)

    time0 = time_ns()
    for i in range(nTime):
        context.execute_async_v3(0)

    time1 = time_ns()
    print(testCase+"average %fms per inference\n"%((time1-time0)/nTime/1000000))


    print("check result:")
    temp1 = bufferH[-1]
    #temp2 = layerNormCPU(bufferH[:1])
    net = CustomLinear(weight,bias)
    temp2 = net(bufferH[0])

    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))),
          "sum diff=%f"%(np.sum(np.abs(temp1 - temp2)))  )

if __name__ == '__main__':

    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)

    run()

    #print("test all finish!")

