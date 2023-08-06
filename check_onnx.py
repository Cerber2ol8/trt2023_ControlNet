from cldm.model import create_model, load_state_dict
import torch
import onnxruntime
import os
import numpy as np
from trt_model import model

model_file = "/home/player/ControlNet/models/control_sd15_canny.pth"
onnx_path = "./onnxsim_model/FrozenCLIPEmbedder.onnx"
trt_path = "./trt_dir/FrozenCLIPEmbedder.engine"

torch_model = create_model('./models/cldm_v15.yaml').cond_stage_model.cpu()


trt_model = model('clip',trt_path)




# model.load_state_dict(
#     load_state_dict(
#         model_file,
#         location='cuda',
#     )
# )


onet_session = onnxruntime.InferenceSession(onnx_path)

input_ids = torch.randint(0,49408,(1, 77),dtype=torch.int32)

with torch.inference_mode():
    outs_torch = torch_model(input_ids).numpy()
    
    outs_onnx = onet_session.run(None, {onet_session.get_inputs()[0].name:input_ids.numpy()})

    outs_trt = trt_model(input_ids)[0]

    

    print(outs_torch)
    print(outs_onnx)
    print(outs_trt)


    abs = np.abs(outs_torch - outs_onnx)
    print(abs.max())

    abs = np.abs(outs_torch - outs_trt)

    print(abs.max())