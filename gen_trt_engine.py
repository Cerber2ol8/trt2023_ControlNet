import os
import tensorrt as trt
import ctypes



onnx_dir = "./onnxsim_model/"
engine_dir = "./trt_dir/"
soFilePath = ["./plugin/LayerNormPlugin/LayerNorm.so","./plugin/CustomLinearPlugin/CustomLinear.so"]
target = "/home/player/ControlNet/plugin/target/"
usePlugin = False


H = 256
W = 384
h = H // 8
w = W // 8

usePlugin =True
verbose = True

logger = trt.Logger(trt.Logger.ERROR)




def build_trt_engine(in_onnx, out_trt,inputs, is_fp16=True):
    #os.system(f"trtexec --onnx={in_onnx} --saveEngine={out_trt} --optShapes={inputs} --{quant} --verbose --buildOnly")
    cmdString = f"trtexec --onnx={in_onnx} --saveEngine={out_trt} --optShapes={inputs} --skipInference --useCudaGraph"
    if is_fp16:
        cmdString += " --fp16"

    if usePlugin:

              
        for path in soFilePath:
            if os.path.exists(path):
                ctypes.cdll.LoadLibrary(path)
                cmdString += " --staticPlugins="  
                cmdString += f"{path} "
        trt.init_libnvinfer_plugins(logger, '')

    if verbose:
        cmdString += " --verbose"  



    os.system(cmdString)


inputs_dict = {"FrozenCLIPEmbedder":"input_ids:1x77",
          "control_net":f"x_in:1x4x{h}x{w},h_in:1x3x{H}x{W},t_in:1,c_in:1x77x768,",
          "unet":f"x_in:1x4x{h}x{w},t_in:1,c_in:1x77x768,",
          "unet_l":f"x_in:1x4x{h}x{w},t_in:1,c_in:1x77x768,",
          "vae_decoder":"z_in:1x4x32x48"}


def get_str(i, shape):
    return f"cl_{i}:{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]},"

for i in range(13):
    if i<3:
        inputs_dict["unet"] += get_str(i,(1, 320, h, w))
    elif i<4:
        inputs_dict["unet"] += get_str(i,(1, 320, h//2, w//2))
    elif i<6:
        inputs_dict["unet"] += get_str(i,(1, 640, h//2, w//2))
    elif i<7:
        inputs_dict["unet"] += get_str(i,(1, 640, h//4, w//4))
    elif i<9:
        inputs_dict["unet"] += get_str(i,(1, 1280, h//4,w//4))
    else:
        inputs_dict["unet"] += get_str(i,(1, 1280, h//8, w//8))
        

for onnx_file in os.listdir(onnx_dir):
    if not os.path.exists(engine_dir):
        os.mkdir(engine_dir)
    if(onnx_file.endswith(".onnx")):
        engine_file = onnx_file.replace(".onnx", ".engine")
        out_file = os.path.join(engine_dir,engine_file)
        if not os.path.exists(out_file):
            model_name = onnx_file.split('.')[0]
            inputs = inputs_dict[model_name]
            if model_name == 'FrozenCLIPEmbedder':
                build_trt_engine(os.path.join(onnx_dir,onnx_file),out_file,inputs,is_fp16=False)
            else:
                build_trt_engine(os.path.join(onnx_dir,onnx_file),out_file,inputs,is_fp16=True)

        else:
            print(f'file {out_file} existed, skip build engine.')

