import os
from typing import Any
import tensorrt as trt
import numpy as np
from cuda import cudart
from annotator.util import resize_image, HWC3
import torch

inputs_dict = {"FrozenCLIPEmbedder":"input_ids:1x77",
          "control_net":"x_in:1x4x32x48,h_in:1x3x256x384,t_in:1,c_in:1x77x768",
          "unet":"x_in:1x4x32x48,t_in:1,c_in:1x77x768",
          "vae_docoder":"z_in:1x4x128x128"}

    # new_img = hk.process(img,
    #         "a bird",
    #         "best quality, extremely detailed",
    #         "longbody, lowres, bad anatomy, bad hands, missing fingers",
    #         1,
    #         256,
    #         20,
    #         False,
    #         1,
    #         9,
    #         2946901,
    #         0.0,
    #         100,
    #         200)
def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    img = resize_image(HWC3(input_image), image_resolution)
    H, W, C = img.shape

#control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)

class model(object):
    def __init__(self, name, engine_file_path) -> None:
        # 创建Logger
        self.trt_logger = trt.Logger()

        self.name = name
        self.engine = self.load_engine(engine_file_path)
        self.context = self.set_context(self.engine)
        self.inputDevice = []
        self.inputHost = []
        self.outputDevice = []
        self.outputHost = []
        self.nInput = 0
        self.nOutput = 0
        self.tensors_input = {}
        self.tensors_output = {}

        self.nIO = self.engine.num_io_tensors


        # 获取IO tensor的名称
        self.lTensorName = [self.engine.get_tensor_name(i) for i in range(self.nIO)]
        self.setIO()



    def setIO(self):
        # 设置各个模型的输入输出
        context = self.context
        name = self.name

        #print(f'----tensor shape of {name} ----')
        # 打印输出模型的tensor shape
        # for t in lTensorName:
        #     print(f'{t} :   {context.get_tensor_shape(t)}')

        H = 256
        W = 384
        h = H // 8
        w = W // 8

        if name == 'clip':
            tensors_input = {"input_ids" :   (1, 77),
                        }

            tensors_output = {"last_hidden_states" :   (1, 77, 768),
                        "pooler_output" :   (1, 768),}

            self.nInput = 1
            self.nOutput = 1


        elif name == 'control':
            self.nInput = 4
            self.nOutput = 13

            #---- input tensor shape of control ----
            tensors_input = {"x_in" :   (1, 4, 32, 48),
                            "h_in" :   (1, 3, 256, 384),
                            "t_in" :   (1,),
                            "c_in" :   (1, 77, 768)}


        elif name == 'unet':
            self.nInput = 16
            self.nOutput = 1


            #---- input tensor shape of unet ----
            tensors_input = {"x_in" :   (1, 4, 32, 48),
                            "t_in" :   (1,),
                            "c_in" :   (1, 77, 768),

                            "cl_0" :   (1, 320, 32, 48),
                            "cl_1" :   (1, 320, 32, 48),
                            "cl_2" :   (1, 320, 32, 48),
                            "cl_3" :   (1, 320, 16, 24),
                            "cl_4" :   (1, 640, 16, 24),
                            "cl_5" :   (1, 640, 16, 24),
                            "cl_6" :   (1, 640, 8, 12),
                            "cl_7" :   (1, 1280, 8, 12),
                            "cl_8" :   (1, 1280, 8, 12),
                            "cl_9" :   (1, 1280, 4, 6),
                            "cl_10" :   (1, 1280, 4, 6),
                            "cl_11" :   (1, 1280, 4, 6),
                            "cl_12" :   (1, 1280, 4, 6)}

        elif name == 'vae':
            self.nInput = 1
            self.nOutput = 1

            #---- input tensor shape of vae ----

            tensors_input ={ "z_in" :   (1, 4, 32, 48)}

        # control_net : torch.Size([1, 4, 32, 48]) torch.Size([1, 3, 256, 384]) torch.Size([1]) torch.Size([1, 77, 768])

        for k,v in tensors_input.items():
            context.set_input_shape(k,v)
        self.tensors_input = tensors_input



    def load_engine(self, engine_file_path):
        assert os.path.exists(engine_file_path)
        trt.init_libnvinfer_plugins(None, "") #初始化插件库
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())


    def set_context(self, engine):
        # 创建上下文
        context = engine.create_execution_context()
        return context
    
    def config_cuda_graph(self):
        cudart.cudaDeviceSynchronize()

        context = self.context
        lTensorName = self.lTensorName
        nInput = self.nInput
        engine = self.engine
        nIO = self.nIO



        # get a CUDA stream for CUDA graph and inference
        _, stream = cudart.cudaStreamCreate()


        bufferD = []
        bufferH = []



        for k,v in self.tensors_input.items():
            data = np.zeros(v,dtype=trt.nptype(engine.get_tensor_dtype(k)))
            bufferH.append(np.ascontiguousarray(data))

        for i in range(nInput, nIO):
            bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

            
        # do inference before CUDA graph capture
        for i in range(nInput):
            cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        for i in range(nIO):
            context.set_tensor_address(lTensorName[i], int(bufferD[i]))

        context.execute_async_v3(stream)

        for i in range(nInput, nIO):
            cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)




        # CUDA Graph capture
        cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        for i in range(nInput):
            cudart.cudaMemcpyAsync(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

        #for i in range(nIO):  # no need to reset the address if unchanged
        #    context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        context.execute_async_v3(stream)
        for i in range(nInput, nIO):
            cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        #cudart.cudaStreamSynchronize(stream)  # no need to synchronize within the CUDA graph capture
        _, graph = cudart.cudaStreamEndCapture(stream)
        cuda_version = int(torch.version.cuda[:2])

        if cuda_version < 12:
            try:
                _, graphExe, _ = cudart.cudaGraphInstantiate(graph, b"", 0)  # for CUDA < 12
            except:
                _, graphExe = cudart.cudaGraphInstantiate(graph, 0)

        else:
            _, graphExe = cudart.cudaGraphInstantiate(graph, 0)  # for CUDA >= 12

        # do inference with CUDA graph
        #bufferH[1] *= 0  # set output buffer as 0 to see the real output of inference
        cudart.cudaGraphLaunch(graphExe, stream)
        cudart.cudaStreamSynchronize(stream)

        print(f"model {self.name} warm up!")
        for i in range(20):
            cudart.cudaGraphLaunch(graphExe, stream)
            cudart.cudaStreamSynchronize(stream)


        self.stream = stream
        self.graphExe = graphExe

        self.bufferD = bufferD
        self.bufferH = bufferH

        # for i in range(nIO):
        #     print(lTensorName[i])
        #     print(bufferH[i])

        # for b in bufferD:
        #     cudart.cudaFree(b)




    def infer_origin(self, inputData, out_gpu = False):
        if self.context != None:
            nInput = self.nInput
            nOutput = self.nOutput

            self.outputHost = []

            # device端申请内存和设置GPU 指针

            for i in range(nInput):

                data = inputData[i]
                # if len(data.shape)>1:
                #     data = data.reshape(-1)

                data = np.ascontiguousarray(data)
                self.inputHost.append(data)
                self.inputDevice.append(cudart.cudaMalloc(data.nbytes)[1])
                self.context.set_tensor_address(self.lTensorName[i], self.inputDevice[i])


            for i in range(nOutput):
                outputHost =  np.empty(self.context.get_tensor_shape(self.lTensorName[nInput+i]),
                                       trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[nInput+i])))
                self.outputHost.append(outputHost)


                self.outputDevice.append(cudart.cudaMalloc(outputHost.nbytes)[1])
                self.context.set_tensor_address(self.lTensorName[nInput+i], self.outputDevice[i])

            # 拷贝数据到device端
            for inputDevice,inputHost in zip(self.inputDevice,self.inputHost):
                cudart.cudaMemcpy(inputDevice, inputHost.ctypes.data, inputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            # 执行推理计算
            self.context.execute_async_v3(0)


            # 释放输入显存
            for inputDevice in self.inputDevice:
                cudart.cudaFree(inputDevice)

            # 拷贝结果到host端
            for outputDevice,outputHost in zip(self.outputDevice,self.outputHost):
                cudart.cudaMemcpy(outputHost.ctypes.data, outputDevice, outputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
                # 释放输出显存
                cudart.cudaFree(outputDevice)



            self.inputDevice = []
            self.outputDevice = []
            self.inputHost = []
            return self.outputHost


    def infer_origin_cuda_graph(self, inputData, out_gpu = False):
        if self.context != None:
            nInput = self.nInput
            nOutput = self.nOutput
            nIO = self.nIO
            context = self.context
            lTensorName = self.lTensorName



            for i in range(nInput):

                data = inputData[i]
                if len(data.shape)>1:
                    data = data.reshape(-1)

                data = np.ascontiguousarray(data)
                #self.bufferH[i].data = data.data

                cudart.cudaMemcpy(self.bufferH[i].ctypes.data, data.ctypes.data, self.bufferH[i].nbytes, 
                                       cudart.cudaMemcpyKind.cudaMemcpyHostToHost)

            
            # # 拷贝数据到device端
            # for i in range(nInput):
                # cudart.cudaMemcpyAsync(self.bufferD[i], self.bufferH[i].ctypes.data, self.bufferH[i].nbytes, 
                #                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,self.stream)

            cudart.cudaGraphLaunch(self.graphExe, self.stream)
            cudart.cudaStreamSynchronize(self.stream)

            return self.bufferH[nInput:]

    def __call__(self, inputData, *args: Any, **kwds: Any) -> Any:
        self.infer_origin(inputData)
        return self.outputHost


class trt_clip(model):
    def __init__(self, name, engine_file_path) -> None:
        super().__init__(name, engine_file_path)

        self.outputData = None

        # 定义输入输出
        self.inputHost0 = np.zeros(self.context.get_tensor_shape(self.lTensorName[0]),
                                trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[0])))
        self.outputHost0 =  np.zeros(self.context.get_tensor_shape(self.lTensorName[1]),
                                trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[1])))
        self.outputHost1 =  np.zeros(self.context.get_tensor_shape(self.lTensorName[2]),
                                trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[2])))

        # device端申请内存
        self.inputDevice = cudart.cudaMalloc(self.inputHost0.nbytes)[1]
        self.outputDevice0 = cudart.cudaMalloc(self.outputHost0.nbytes)[1]
        self.outputDevice1 = cudart.cudaMalloc(self.outputHost1.nbytes)[1]

        # 绑定device端地址
        self.context.set_tensor_address(self.lTensorName[0], self.inputDevice)
        self.context.set_tensor_address(self.lTensorName[1], self.outputDevice0)
        self.context.set_tensor_address(self.lTensorName[2], self.outputDevice1)


    def infer(self, input_ids : np.array):

        input_ids = np.ascontiguousarray(input_ids)
        cudart.cudaMemcpy(self.inputDevice, input_ids.ctypes.data, input_ids.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # 执行推理计算
        self.context.execute_async_v3(0)

        cudart.cudaMemcpy(self.outputHost0.ctypes.data, self.outputDevice0, self.outputHost0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        #self.cudaFree()

        return self.outputHost0

    def __call__(self, input_ids : np.array):

        return self.infer(input_ids=input_ids)


    def cudaFree(self):
        # 释放device内存

        if self.inputDevice != None:
            cudart.cudaFree(self.inputDevice)

        if self.outputDevice0 != None:
            cudart.cudaFree(self.outputDevice0)

        if self.outputDevice1 != None:
            cudart.cudaFree(self.outputDevice1)





class trt_control_net(model):
    def __init__(self, name, engine_file_path) -> None:
        super().__init__(name, engine_file_path)




    def infer(self, inputData):
        if self.context != None:
            nInput = self.nInput
            nOutput = self.nOutput

            self.outputHost = []

            # device端申请内存和设置GPU 指针

            for i in range(nInput):

                data = inputData[i]
                #如果是地址
                if(i == 1 or i == 3):
                    self.inputHost.append(data)
                    self.inputDevice.append(data)
                    self.context.set_tensor_address(self.lTensorName[i], self.inputDevice[i])
                else:
                    if len(data.shape)>1:
                        data = data.reshape(-1)

                    data = np.ascontiguousarray(data)
                    self.inputHost.append(data)
                    self.inputDevice.append(cudart.cudaMalloc(data.nbytes)[1])
                    self.context.set_tensor_address(self.lTensorName[i], self.inputDevice[i])


            for i in range(nOutput):
                outputHost =  np.empty(self.context.get_tensor_shape(self.lTensorName[nInput+i]),
                                       trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[nInput+i])))
                self.outputHost.append(outputHost)


                self.outputDevice.append(cudart.cudaMalloc(outputHost.nbytes)[1])
                self.context.set_tensor_address(self.lTensorName[nInput+i], self.outputDevice[i])

            # 拷贝数据到device端

            for i in range(len(self.inputDevice)):
                if(i != 1 and i != 3):
                    cudart.cudaMemcpy(self.inputDevice[i], self.inputHost[i].ctypes.data, self.inputHost[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            # 执行推理计算
            self.context.execute_async_v3(0)


            # 释放输入显存
            for i in range(len(self.inputDevice)):
                if(i != 1 and i != 3):
                    cudart.cudaFree(self.inputDevice[i])

            # 拷贝结果到host端
            for outputDevice,outputHost in zip(self.outputDevice,self.outputHost):
                cudart.cudaMemcpy(outputHost.ctypes.data, outputDevice, outputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
                # 释放输出显存
                cudart.cudaFree(outputDevice)



            self.inputDevice = []
            self.outputDevice = []
            self.inputHost = []
            return self.outputHost


    def freeCuda_output(self):
        # 释放输出显存
        for i in range(len(self.outputDevice)):
            cudart.cudaFree(self.outputDevice[i])
        self.outputDevice.clear()
        self.outputHost.clear()

    def freeCuda_input(self):
        # 释放输入显存
        for i in range(len(self.inputDevice)):
            cudart.cudaFree(self.inputDevice[i])
        self.inputDevice.clear()
        self.inputHost.clear()

    def freeCuda(self):

        cudart.cudaFree(self.noisyDevice)
        cudart.cudaFree(self.condDevice)
        cudart.cudaFree(self.tDevice)
        cudart.cudaFree(self.hintDevice)
        # cudart.cudaFree(self.ucondDevice)
        # cudart.cudaFree(self.uhintDevice)
        for i in range(len(self.inputDevice)):
            cudart.cudaFree(self.inputDevice[i])
        self.inputDevice.clear()
        self.freeCuda_output()
        self.is_malloc = False


    def __call__(self, inputData, *args: Any, **kwds: Any):
        return self.infer(inputData)


class trt_unet(model):
    def __init__(self, name, engine_file_path) -> None:
        super().__init__(name, engine_file_path)


    def infer_origin(self, control_input):

        # 拷贝input数据到device
        # cudart.cudaMemcpy(self.noisyDevice, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        # cudart.cudaMemcpy(self.tDevice, t.ctypes.data, t.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        # cudart.cudaMemcpy(self.condDevice, c.ctypes.data, c.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(self.nInput):
            data = np.ascontiguousarray(control_input[i])
            cudart.cudaMemcpy(self.inputDevice[i], data.ctypes.data, data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # 执行推理计算
        self.context.execute_async_v3(0)


        # 拷贝结果到host端
        cudart.cudaMemcpy(self.erpHost.ctypes.data, self.erpDevice, self.erpHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return self.erpHost

    def infer(self, inputData):
        if self.context != None:
            nInput = self.nInput
            nOutput = self.nOutput

            self.outputHost = []

            # device端申请内存和设置GPU 指针

            for i in range(nInput):

                data = inputData[i]
                #如果是地址
                if(i == 2):
                    self.inputDevice.append(data)
                    self.inputHost.append(data)
                    self.context.set_tensor_address(self.lTensorName[i], self.inputDevice[i])
                else:
                    if len(data.shape)>1:
                        data = data.reshape(-1)

                    data = np.ascontiguousarray(data)
                    self.inputHost.append(data)
                    self.inputDevice.append(cudart.cudaMalloc(data.nbytes)[1])
                    self.context.set_tensor_address(self.lTensorName[i], self.inputDevice[i])


            for i in range(nOutput):
                outputHost =  np.empty(self.context.get_tensor_shape(self.lTensorName[nInput+i]),
                                       trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[nInput+i])))
                self.outputHost.append(outputHost)


                self.outputDevice.append(cudart.cudaMalloc(outputHost.nbytes)[1])
                self.context.set_tensor_address(self.lTensorName[nInput+i], self.outputDevice[i])

            # 拷贝数据到device端
            for i in range(len(self.inputDevice)):
                if(i != 2):
                    cudart.cudaMemcpy(self.inputDevice[i], self.inputHost[i].ctypes.data, self.inputHost[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            # 执行推理计算
            self.context.execute_async_v3(0)


            # 释放输入显存
            for i in range(len(self.inputDevice)):
                if(i != 2):
                    cudart.cudaFree(self.inputDevice[i])

            # 拷贝结果到host端
            for outputDevice,outputHost in zip(self.outputDevice,self.outputHost):
                cudart.cudaMemcpy(outputHost.ctypes.data, outputDevice, outputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
                # 释放输出显存
                cudart.cudaFree(outputDevice)



            self.inputDevice = []
            self.outputDevice = []
            self.inputHost = []
            return self.outputHost




    def __call__(self, control_input, *args: Any, **kwds: Any) -> Any:
        return self.infer(control_input)

    def freeCuda(self):
        cudart.cudaFree(self.erpDevice)

class trt_engine():

    def __init__(self, path_to_engine="./trt_dir/") -> None:
        import ctypes

        
        soFilePath = ["./plugin/target/LayerNorm.so",
                      "./plugin/target/CustomLinear.so"]
        for path in soFilePath:
            if os.path.exists(path):
                ctypes.cdll.LoadLibrary(path)

        

        # 创建engine
        self.clip = model('clip', os.path.join(path_to_engine,"FrozenCLIPEmbedder.engine"))
        #self.clip = None
        self.control = model('control', os.path.join(path_to_engine,"control_net.engine"))
        #self.unet = trt_unet('unet', os.path.join(path_to_engine,"unet.engine"),self.control)
        self.unet = model('unet', os.path.join(path_to_engine,"unet.engine"))
        self.vae = model('vae', os.path.join(path_to_engine,"vae_decoder.engine"))
        #self.vae = None



# if __name__ == '__main__':
#     engines = trt_engine()

#     clip = engines.clip
#     input_ids = np.random.randint(0,49408,(3, 77))
#     clip.infer(input_ids)

