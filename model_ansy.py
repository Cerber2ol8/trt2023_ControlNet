from canny2image_TRT_old import hackathon




# 取4个子模型，做一个映射
state_dict = {
    "clip": "cond_stage_model",
    "control_net": "control_model",
    "unet": "diffusion_model",
    "vae": "first_stage_model"
}

if __name__ == '__main__':

    hk = hackathon()
    hk.initialize()
    unet = hk.model.model.diffusion_model
    print(unet)