import os
import tensorrt as trt
import numpy as np
from cuda import cudart


TRT_LOGGER = trt.Logger()
engine_file = ""
engine_path = "./trt_dir/"


'''

----tensor shape of clip ----
input_ids :   (1, 77)
last_hidden_states :   (1, 77, 768)
pooler_output :   (1, 768)
----tensor shape of contorl ----
x_in :   (1, 4, 32, 48)
h_in :   (1, 3, 256, 384)
t_in :   (1,)
c_in :   (1, 77, 768)
cl_out_0 :   (1, 320, 32, 48)
cl_out_1 :   (1, 320, 32, 48)
cl_out_2 :   (1, 320, 32, 48)
cl_out_3 :   (1, 320, 16, 24)
cl_out_4 :   (1, 640, 16, 24)
cl_out_5 :   (1, 640, 16, 24)
cl_out_6 :   (1, 640, 8, 12)
cl_out_7 :   (1, 1280, 8, 12)
cl_out_8 :   (1, 1280, 8, 12)
cl_out_9 :   (1, 1280, 4, 6)
cl_out_10 :   (1, 1280, 4, 6)
cl_out_11 :   (1, 1280, 4, 6)
cl_out_12 :   (1, 1280, 4, 6)
----tensor shape of unet ----
x_in :   (1, 4, 32, 48)
t_in :   (1,)
c_in :   (1, 77, 768)
cl_0 :   (1, 320, 32, 48)
cl_1 :   (1, 320, 32, 48)
cl_2 :   (1, 320, 32, 48)
cl_3 :   (1, 320, 16, 24)
cl_4 :   (1, 640, 16, 24)
cl_5 :   (1, 640, 16, 24)
cl_6 :   (1, 640, 8, 12)
cl_7 :   (1, 1280, 8, 12)
cl_8 :   (1, 1280, 8, 12)
cl_9 :   (1, 1280, 4, 6)
cl_10 :   (1, 1280, 4, 6)
cl_11 :   (1, 1280, 4, 6)
cl_12 :   (1, 1280, 4, 6)
unet_output :   (1, 4, 32, 48)
----tensor shape of vae ----
z_in :   (1, 4, 1, 1)
vae_out :   (1, 3, 8, 8)
'''

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    

def infer():
    engine = load_engine(engine_file)

    pass

def inspect(engine):
    inspector = engine.create_engine_inspector()
    # inspector.execution_context = context; # OPTIONAL
    # print(inspector.get_layer_information(0, LayerInformationFormat.JSON); # Print the information of the first layer in the engine.
    # print(inspector.get_engine_information(LayerInformationFormat.JSON); # Print the information of the entire engine.


def main():
    # 生成engine
    trt.init_libnvinfer_plugins(None, "") #初始化插件库
    engine = load_engine(os.path.join(engine_path,"unet.engine"))

    # print("输入",engine.get_binding_shape(0))
    # print("输出",engine.get_binding_shape(1))

    nIO = engine.num_io_tensors
    # 获取IO tensor的名称
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]

    # 创建上下文
    context = engine.create_execution_context()

    
    for t in lTensorName:
        print(t)
        print(context.get_tensor_shape(t))

    exit()

    # 定义输入尺寸 clip: [1,77]
    # control_net : torch.Size([1, 4, 32, 48]) torch.Size([1, 3, 256, 384]) torch.Size([1]) torch.Size([1, 77, 768])
    inputShape = [1,77]

    # 定义输出尺寸 clip: [1, 77, 768] [1, 768] 
    # outputShape0 = [1, 77, 768]
    # outputShape1 = [1, 768]


    context.set_input_shape(lTensorName[0], inputShape)
    #print(lTensorName)


    #inputData = np.random.randint(0,49408,(3, 77))
    inputData = []
    inputData.append(np.random.random((1,4,32,48)))

    print(inputData)

    inputHost = np.ascontiguousarray(inputData.reshape(-1))

    outputHost0 = np.empty(context.get_tensor_shape(lTensorName[1]), trt.nptype(engine.get_tensor_dtype(lTensorName[1])))

    outputHost1 = np.empty(context.get_tensor_shape(lTensorName[2]), trt.nptype(engine.get_tensor_dtype(lTensorName[2])))

    # device端申请内存
    inputDevice = cudart.cudaMalloc(inputHost.nbytes)[1]
    outputDevice0 = cudart.cudaMalloc(outputHost0.nbytes)[1]
    outputDevice1 = cudart.cudaMalloc(outputHost1.nbytes)[1]

    ## 用到的 GPU 指针提前在这里设置，不再传入 execute_v3 函数
    context.set_tensor_address(lTensorName[0], inputDevice) 
    context.set_tensor_address(lTensorName[1], outputDevice0)
    context.set_tensor_address(lTensorName[2], outputDevice1)


    # 拷贝数据到device端
    cudart.cudaMemcpy(inputDevice, inputHost.ctypes.data, inputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    # 执行计算
    context.execute_async_v3(0) 

    # 拷贝结果到host端
    cudart.cudaMemcpy(outputHost0.ctypes.data, outputDevice0, outputHost0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaMemcpy(outputHost1.ctypes.data, outputDevice1, outputHost1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print(outputHost0,outputHost1)


    pass

if __name__ == '__main__':
    main()