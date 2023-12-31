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
            self.nOutput = 2


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
    

    
    def infer(self, inputData, out_gpu = False):
        if self.context != None:
            nInput = self.nInput
            nOutput = self.nOutput

            self.outputHost = []
            
            # device端申请内存和设置GPU 指针

            for i in range(nInput):
                
                data = inputData[i]
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
            

    def __call__(self, inputData, *args: Any, **kwds: Any) -> Any:
        self.infer(inputData)
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

        self.cond = None
        self.ucond = None
        self.loaded = False
        self.is_malloc = False

        # 定义输入 
        self.noisyHost = np.zeros(self.context.get_tensor_shape(self.lTensorName[0]), 
                            trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[0])))
        self.hintHost = np.zeros(self.context.get_tensor_shape(self.lTensorName[1]), 
                            trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[1])))
        self.tHost = np.zeros(self.context.get_tensor_shape(self.lTensorName[2]), 
                            trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[2])))
        self.condHost = np.zeros(self.context.get_tensor_shape(self.lTensorName[3]), 
                            trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[3])))
    
        self.noisyDevice = cudart.cudaMalloc(self.noisyHost.nbytes)[1]
        self.hintDevice = cudart.cudaMalloc(self.hintHost.nbytes)[1]
        self.tDevice = cudart.cudaMalloc(self.tHost.nbytes)[1]
        self.condDevice = cudart.cudaMalloc(self.condHost.nbytes)[1]

        # 绑定内存地址
        self.context.set_tensor_address(self.lTensorName[0], self.noisyDevice) 
        self.context.set_tensor_address(self.lTensorName[1], self.hintDevice) 
        self.context.set_tensor_address(self.lTensorName[2], self.tDevice) 
        self.context.set_tensor_address(self.lTensorName[3], self.condDevice) 

        # 定义输出 
        for i in range(self.nOutput):
            outputHost = np.empty(self.context.get_tensor_shape(self.lTensorName[self.nInput+i]), 
                                    trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[self.nInput+i])))
            self.outputHost.append(outputHost)

            self.outputDevice.append(cudart.cudaMalloc(self.outputHost[i].nbytes)[1])
            self.context.set_tensor_address(self.lTensorName[self.nInput+i], self.outputDevice[i]) 

    def infer_origin(self, x_noisy, hint, t, cond_txt):


        # 拷贝输入数据到device端
        cudart.cudaMemcpy(self.noisyDevice, x_noisy.ctypes.data, x_noisy.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.hintDevice, hint.ctypes.data, hint.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.tDevice, t.ctypes.data, t.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.condDevice, cond_txt.ctypes.data, cond_txt.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # 执行推理计算
        self.context.execute_async_v3(0)

        # 拷贝结果到host端
        for outputDevice,outputHost in zip(self.outputDevice,self.outputHost):
            cudart.cudaMemcpy(outputHost.ctypes.data, outputDevice, outputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            # 等采样结束后再释放显存
            # cudart.cudaFree(outputDevice)


        return self.outputHost
    def infer(self, x_noisy, hint, t, cond_txt):


        # 拷贝输入数据到device端
        cudart.cudaMemcpy(self.noisyDevice, x_noisy.ctypes.data, x_noisy.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        #cudart.cudaMemcpy(self.hintDevice, hint.ctypes.data, hint.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self.hintDevice = hint
        self.context.set_tensor_address(self.lTensorName[1], self.hintDevice) 
        cudart.cudaMemcpy(self.tDevice, t.ctypes.data, t.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        #cudart.cudaMemcpy(self.condDevice, cond_txt.ctypes.data, cond_txt.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self.condDevice = cond_txt
        self.context.set_tensor_address(self.lTensorName[3], self.condDevice) 

        # 执行推理计算
        self.context.execute_async_v3(0)

        # 拷贝结果到host端
        for outputDevice,outputHost in zip(self.outputDevice,self.outputHost):
            cudart.cudaMemcpy(outputHost.ctypes.data, outputDevice, outputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            # 等采样结束后再释放显存
            # cudart.cudaFree(outputDevice)


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


    def __call__(self,  x_noisy, hint, t, cond_txt, *args: Any, **kwds: Any):
        return self.infer( x_noisy, hint, t, cond_txt)


class trt_unet(model):
    def __init__(self, name, engine_file_path) -> None:
        super().__init__(name, engine_file_path)

        for i in range(self.nInput):
            data = np.empty(self.context.get_tensor_shape(self.lTensorName[i]), 
                            trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i])))
            data = np.ascontiguousarray(data)
            self.inputHost.append(data)
            self.inputDevice.append(cudart.cudaMalloc(data.nbytes)[1])
            self.context.set_tensor_address(self.lTensorName[i], self.inputDevice[i]) 


        # 输出
        self.erpHost = np.empty(self.context.get_tensor_shape(self.lTensorName[16]), 
                            trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[16])))
        self.erpDevice = cudart.cudaMalloc(self.erpHost.nbytes)[1]
        self.context.set_tensor_address(self.lTensorName[self.nInput], self.erpDevice)

        

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

    def infer(self, control_input):

        # 拷贝input数据到device
        # cudart.cudaMemcpy(self.noisyDevice, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        # cudart.cudaMemcpy(self.tDevice, t.ctypes.data, t.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        # cudart.cudaMemcpy(self.condDevice, c.ctypes.data, c.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        for i in range(self.nInput):
            if i == 2:
                # 已经有的数据 cond_txt 不进行拷贝
                cudart.cudaFree(self.inputDevice[i])
                self.inputDevice[i] = control_input[i]
                self.context.set_tensor_address(self.lTensorName[i], self.inputDevice[i]) 
            else:
                data = np.ascontiguousarray(control_input[i])
                cudart.cudaMemcpy(self.inputDevice[i], data.ctypes.data, data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # 执行推理计算
        self.context.execute_async_v3(0)


        # 拷贝结果到host端
        cudart.cudaMemcpy(self.erpHost.ctypes.data, self.erpDevice, self.erpHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return self.erpHost


    def infer1(self, control_input ,is_cond):
        
        #assert self.control.loaded == True
        if is_cond:
            self.context.set_tensor_address(self.lTensorName[2], self.condDevice) 
        else:
            self.context.set_tensor_address(self.lTensorName[2], self.ucondDevice) 

        # 拷贝输入
        assert len(control_input) == 13
        for i in range(13):
            data = control_input[i]
            if torch.is_tensor(data):
                data = data.cpu().numpy()
            data = np.ascontiguousarray(data)
            cudart.cudaMemcpy(self.control_inputDevice[i], data.ctypes.data, data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # 执行推理计算
        self.context.execute_async_v3(0)

        # 拷贝输出
        cudart.cudaMemcpy(self.erpHost.ctypes.data, self.erpDevice, self.erpHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        return self.erpHost


    
    def __call__(self, control_input, is_cond, *args: Any, **kwds: Any) -> Any:
        return self.infer(control_input, is_cond)
    
    def freeCuda(self):
        cudart.cudaFree(self.erpDevice)

class trt_engine():

    def __init__(self, path_to_engine="./trt_dir/") -> None:


        # 创建engine
        self.clip = model('clip', os.path.join(path_to_engine,"FrozenCLIPEmbedder.engine"))
        self.control = trt_control_net('control', os.path.join(path_to_engine,"control_net.engine"))
        #self.unet = trt_unet('unet', os.path.join(path_to_engine,"unet.engine"),self.control)
        self.unet = trt_unet('unet', os.path.join(path_to_engine,"unet.engine"))
        self.vae = model('vae', os.path.join(path_to_engine,"vae_decoder.engine"))




# if __name__ == '__main__':
#     engines = trt_engine()

#     clip = engines.clip
#     input_ids = np.random.randint(0,49408,(3, 77))
#     clip.infer(input_ids)

