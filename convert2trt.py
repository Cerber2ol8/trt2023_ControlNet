import os
import ctypes
import onnx
import argparse
import numpy as np
import tensorrt as trt
import onnx_graphsurgeon as gs
#from calibrator import MobileVitCalibrator


unet_input = {"x_in" :   (1, 4, 32, 48),
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

control_input = {"x_in" :   (1, 4, 32, 48),
                "h_in" :   (1, 3, 256, 384),
                "t_in" :   (1,),
                "c_in" :   (1, 77, 768)}

vae_input ={ "z_in" :   (1, 4, 32, 48)}


clip_input = {"input_ids" :   (1, 77)}


def trt_builder_plugin(onnxFile,trtFile,in_shapes,workspace=22,pluginFileList=[],
                       use_fp16=False,set_int8_precision=False,verbose=False,optimization=3,steams=-1):
    
    if os.path.exists(trtFile):
        print("engine exists, skip build.")
        exit()

    if verbose:
        logger = trt.Logger(trt.Logger.VERBOSE)#ERROR INFO  VERBOSE
    else:
        logger = trt.Logger()

    trt.init_libnvinfer_plugins(logger, '')
    if len(pluginFileList)>0:
        for pluginFile in pluginFileList:
            ctypes.cdll.LoadLibrary(pluginFile)
            print("load plugin",pluginFile)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.builder_optimization_level = optimization

    if steams > 0:
        config.max_aux_streams = steams


    profile = builder.create_optimization_profile()
    config.max_workspace_size = (1 << 30)*workspace
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")
    for i in range(network.num_inputs):
        inputTensor = network.get_input(i)
        name=inputTensor.name
        if name in in_shapes:
            #profile.set_shape(name, in_shapes[name][0],in_shapes[name][1],in_shapes[name][2])
            profile.set_shape(name, in_shapes[name],in_shapes[name],in_shapes[name])
        
    config.add_optimization_profile(profile)
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if set_int8_precision:
        config.set_flag(trt.BuilderFlag.INT8)
        #config.int8_calibrator=MobileVitCalibrator()
        config.set_calibration_profile(profile)     

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if "LayerNorm" in layer.name:
            layer.precision = trt.float32
            print("reset precision to FP32: ",layer.name)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)
        print("Succeeded save engine!")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='onnx convert to trt describe.')


    parser.add_argument(
        "--model_name",
        type = str,
        default="unet")

    parser.add_argument(
        "--input_dir",
        type = str,
        default="onnxsim_model")

    parser.add_argument(
        "--save_dir",
        type=str,
        default="trt_dir")
        
    parser.add_argument(
        "--dynamic",
        default=False, action='store_true',
        help="export  dynamic onnx model , default is True.")
        
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="batchsize of onnx models, default is 1.")

    parser.add_argument(
        "--fp16",
        default=False, action='store_true',
        help="use fp16, default is False.")
        
    parser.add_argument(
        "--int8", action='store_true',
        help="use int8 , default is False.")
    
    parser.add_argument(
        "--use_plugins", default=False, action='store_true',
        help="use plugins , default is False.")
    
    parser.add_argument(
        "--verbose", default=False, action='store_true',
        help="verbose info, default is False.")
    
    parser.add_argument(
        "--optim_level", 
        type=int,
        help="trt build optimization level, 0: build fast but optimization less, 5: build slow but optimization better, default is 3.")
    

    parser.add_argument(
    "--max_aux_streams", 
    type=int, default=-1,
    help="trt build optimization level, 0: build fast but optimization less, 5: build slow but optimization better, default is 3.")


    args = parser.parse_args()
    print(args)


    source = os.path.join(args.input_dir, args.model_name + ".onnx")
    target = os.path.join(args.save_dir, args.model_name + ".engine")


    inputs = {}
    plugins = []

    if args.use_plugins:
        plugins = [os.path.join( "plugin/target", filename) for filename in os.listdir("plugin/target")]


    if args.model_name == "unet":
        inputs = unet_input
    elif args.model_name == "control_net":
        inputs = control_input
    elif args.model_name == "vae_decoder":
        inputs = vae_input
    elif args.model_name == "FrozenCLIPEmbedder":
        inputs = clip_input


    #   if args.dynamic:
    #     encoder_in_shapes={'input':[(1,77,320),(4,3,256,256),(8,3,256,256)]}
    #   else:
    #     encoder_in_shapes={'input':[(args.batch,3,256,256),(args.batch,3,256,256),(args.batch,3,256,256)]}
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)



    trt_builder_plugin(source,target,
                        inputs,pluginFileList=plugins,
                        use_fp16=args.fp16,set_int8_precision=args.int8,verbose=args.verbose,optimization=args.optim_level)
