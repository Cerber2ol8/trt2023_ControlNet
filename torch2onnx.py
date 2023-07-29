# -*- coding: utf-8 -*-
import torch
import onnx
import os
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
import gc

model_file = '/home/player/ControlNet/models/control_sd15_canny.pth'
onnx_path = './onnx_models/'

if not torch.cuda.is_available() :
    print("cuda is not avilable!")
    exit()

if not os.path.exists(onnx_path):
    os.mkdir(onnx_path)

files = os.listdir(onnx_path)




def model():
    model = create_model(model_file).cpu()
    model.load_state_dict(
        load_state_dict(
            model_file,
            location='cuda',
        )
    )
    model = model.cuda()
    return model


def print_state_dict(state_dict):    
    for layer in state_dict:
        print(layer, '\t', state_dict[layer].shape)
        if layer.find("input")>-1:
            print("find input",layer)

def print_models(state_dict):
    for k in state_dict.items():
        print(k)


def get_onnx(model):

    # 取4个子模型，做一个映射
    state_dict = {
        "clip": "cond_stage_model",
        "control_net": "control_model",
        "unet": "diffusion_model",
        "vae": "first_stage_model"
    }

    # for layer in state_dict:
    #     print(layer, '\t', state_dict[layer].shape)
    #     if layer.find("input")>-1:
    #         print("find input",layer)
    H = 256
    W = 384

    for k, v in state_dict.items():
        if k != "unet":
            temp_model = getattr(model, v)
        else:
            # unet比较特殊，在model.model下面的一个diffusion_model属性里面
            temp_model = getattr(model.model, v)

        # temp_model为对应的子模型
        # 由于onnx导出的时候一般是导出__call__函数，而__call__这个函数在torch.nn.Module的类下面一般会调用forward函数
        # 有两个在pipline中调用的不是forward函数，所以下面要做一些函数替换
        # 先导出clip
        if k == "clip":


            
            # clip调用了encode,encode调用了forward
            # forward输入时，一共就两步，一个是tokenizer,一个是transformer
            # transformer的__call__默认也是forward,所以干脆我直接去clip的transformer导出onnx就行了。
            _model = temp_model.transformer
            # 这里把tokenizer单独拿出来做一个函数，后续给prompt做分词用
            # tokenizer = temp_model.tokenizer
            # tokens = tokenizer.tokenize("this is a test sequence")
            # ids = tokenizer.convert_tokens_to_ids(tokens)
            
            #exit()
            
            # 然后下面是onnx导出代码
            #input_ids = torch.tensor([ids]).cuda()


            # 下面的inputs是输入变量列表，你可以去debug一下，实际是一个[torch.Tensor]
            #print(_model.forward)
            input_ids = torch.randint(0,49408,(3, 77)).to("cuda")
            #output = _model(input_ids)
            # onnx_path: 保存路径
            # input_names, output_names 对应输入输出名列表，这个debug一下，可以随便命名
            # dynamic_axes是一个dict，这里我给一个样例：{'input_ids': {0: 'B'},'text_embeddings': {0: 'B'}}
            # 大概意思就是输入变量`input_ids`和输出变量`text_embeddings`的第0维可以动态变化，其实也就是batch_size支持动态咯。
            # opset_version: 这里固定为17就行

            #text_embeddings = _model(input_ids)

            
            out = "FrozenCLIPEmbedder.onnx"
            if out in files:
                continue
            print("exporting FrozenCLIPEmbedder...")
            torch.onnx.export(
                _model,
                (input_ids),
                onnx_path+out,
                export_params=True,
                verbose=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["last_hidden_states","pooler_output"],
                dynamic_axes={'input_ids': {0: 'bs'},'last_hidden_states': {0: 'bs'},'pooler_output': {0: 'bs'}},
            )


        # 再导出control_net
        elif k == "control_net":
            out = "control_net.onnx"




            _model = temp_model
            x_in = torch.randn(1,4,H//8, W //8, dtype=torch.float32).to("cuda")
            h_in = torch.randn(1,3,H, W, dtype=torch.float32).to("cuda")
            t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
            c_in = torch.randn(1,77,768, dtype=torch.float32).to("cuda")



            #input_names = ['rand_input','hints','timesteps','ids']
            #output_list = [f'cl_output_{i}' for i in range(13)]

            output_names = []
            for i in range(13):
                output_names.append("cl_out_"+ str(i))

            dynamic_table = {'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
                                'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
                                't_in' : {0 : 'bs'},
                                'c_in' : {0 : 'bs'}}
            for i in range(13):
                dynamic_table[output_names[i]] = {0 : "bs"}

            #output = _model(x_in, h_in, t_in, c_in)
            # for t in output:
            #     print(t.shape)
            # torch.Size([1, 320, 32, 48])
            # torch.Size([1, 320, 32, 48])
            # torch.Size([1, 320, 32, 48])
            # torch.Size([1, 320, 16, 24])
            # torch.Size([1, 640, 16, 24])
            # torch.Size([1, 640, 16, 24])
            # torch.Size([1, 640, 8, 12])
            # torch.Size([1, 1280, 8, 12])
            # torch.Size([1, 1280, 8, 12])
            # torch.Size([1, 1280, 4, 6])
            # torch.Size([1, 1280, 4, 6])
            # torch.Size([1, 1280, 4, 6])
            # torch.Size([1, 1280, 4, 6])
            if out in files:
                continue
            print("exporting control_net ...")

            torch.onnx.export(
                _model,
                (x_in, h_in, t_in, c_in),
                onnx_path+out,
                export_params=True,
                verbose=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['x_in', "h_in", "t_in", "c_in"],
                output_names=output_names,
                dynamic_axes=dynamic_table
            )

        # 导出unet
        elif k == "unet":
            out = "unet.onnx"

            _model = temp_model

            h = H // 8
            w = W // 8



            x_in = torch.randn(1,4,h,w, dtype=torch.float32).to("cuda")
            t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
            c_in = torch.randn(1,77,768, dtype=torch.float32).to("cuda")

            inputs = []
            inputs.append(x_in)
            inputs.append(t_in)
            inputs.append(c_in)
            #inputs.append(control)
            

            input_names = ["x_in","t_in","c_in"]
            #input_names.append('control')

            dynamic_table = {'x_in' : {0 : 'bs', 2 : 'x_H', 3 : 'x_W'}, 
                                't_in' : {0 : 'bs'},
                                'c_in' : {0 : 'bs'},}
                                #'control' : {0 : 'bs', 1 : 'C', 2 : 'H', 3 : 'W'}}
            
            #output = _model(x_in,t_in,c_in,control) # [1,4,32,48]
            input_cldm = []

            for i in range(13):
                if i<3:
                    input_cldm.append(torch.randn((1, 320, h, w)).cuda())
                elif i<4:
                    input_cldm.append(torch.randn((1, 320, h//2, w//2)).cuda())
                elif i<6:
                    input_cldm.append(torch.randn((1, 640, h//2, w//2)).cuda())
                elif i<7:
                    input_cldm.append(torch.randn((1, 640, h//4, w//4)).cuda())
                elif i<9:
                    input_cldm.append(torch.randn((1, 1280, h//4, w//4)).cuda())
                else:
                    input_cldm.append(torch.randn((1, 1280, h//8, w//8)).cuda())

            #forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs)
            control = input_cldm.reverse()
 
            
            for i in range(13):
                dynamic_table[f'cl_{i}'] = {0 : 'bs', 2 : f'cl_H{i}', 3 : f'cl_W{i}'}
                input_names.append(f'cl_{i}')
                inputs.append(input_cldm.pop())


            dynamic_table['unet_output'] = {0 : 'bs'}



            if out in files:
                continue
            print("exporting unet ...")
            torch.onnx.export(
                _model,
                tuple(inputs),
                f=onnx_path+out,
                export_params=True,
                #verbose=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=input_names,
                output_names=["unet_output"],
                dynamic_axes=dynamic_table
            )

            # 和control_net一样的操作，一样的bug,就是输入输出有些不一样
            # unet输入包含controlNet的输出，所以input_names里面。由于controlNet的输出是一个长度为13的List，所以这个input_names,需要将对应变量的输入名，改成13个，例如control_1, control_2..control_13这样。
        # 再导出vae
        elif k == "vae":
            out = "vae_decoder.onnx"

            _model = temp_model
            # vae调用的是decode,而导出onnx需要forward,所以这里做一个替换即可。
            _model.forward = _model.decode
            z_in = torch.randn(1,4,128,128, dtype=torch.float32).to("cuda")
            #outputs = _model(z_in) #[1,4,256,256]

            if out in files:
                continue
            print("exporting vae decoder...")
            # 然后和上面一样，做onnx导出
            torch.onnx.export(
                _model,
                z_in,
                onnx_path+out,
                export_params=True,
                verbose=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["z_in"],
                output_names=["vae_out"],
                dynamic_axes={'z_in': {0: 'B', 2 : 'H', 3 : 'W'}, 'vae_out': {0: 'B'}},
            )
        else:
            _model = None
        # 导onnx后，这里model其实可以丢掉了，防止他占用显存和内存
        if _model is not None:
            del _model
            torch.cuda.empty_cache()
            gc.collect()

def troch2onnx(model):




    # 构建输入数据
    x = torch.randn(1, 1, 224, 224, requires_grad=True)
    # 导出模型
    torch.onnx.export(model,               # 要转换的模型
                    x,                         # 模型输入
                    onnx_path,   # 模型保存位置              
                    opset_version=11,          # onnx算子版本                
                    input_names = ['input'],   # 模型的输入名称
                    output_names = ['output']  # 模型的输出名称
                    )


model = model()

# 打印输出模型
#print(model)
#print(model.first_stage_model)
#print_state_dict(model)

# 设置模型为推理状态
#model.eval()

get_onnx(model)
