
# 保存 data_loader生成的输入和运行得到的结果
polygraphy run onnxsim_model/unet.onnx  --onnxrt  --val-range [0,1] \
--save-inputs  data/net_input.json --save-outputs data/onnx_res.json \
--onnx-outputs mark all \
--input-shapes x_in:[2,4,32,48] \
t_in:[2,] \
c_in:[2,77,768] \
cl_0:[2,320,32,48] \
cl_1:[2,320,32,48] \
cl_2:[2,320,32,48] \
cl_3:[2,320,16,24] \
cl_4:[2,640,16,24] \
cl_5:[2,640,16,24] \
cl_6:[2,640,8,12] \
cl_7:[2,1280,8,12] \
cl_8:[2,1280,8,12] \
cl_9:[2,1280,4,6] \
cl_10:[2,1280,4,6] \
cl_11:[2,1280,4,6] \
cl_12:[2,1280,4,6]


# 运行trt engine，并与onnx推理得到的结果作比较
polygraphy run trt_dir/unet_fp32_o3_b2.engine --model-type engine --trt --load-outputs data/onnx_res.json --abs 1e-4 \
--trt-outputs mark all \
--load-inputs  data/net_input.json \
--fail-fast




# 运行trt engine，并与onnx推理得到的结果作比较
polygraphy run onnxsim_model/unet.onnx  --trt --onnxrt \
--abs 1e-4 \
--onnx-outputs mark all \
--trt-outputs mark all \
--load-inputs  data/net_input.json \
--fail-fast \
--input-shapes x_in:[2,4,32,48] \
t_in:[2,] \
c_in:[2,77,768] \
cl_0:[2,320,32,48] \
cl_1:[2,320,32,48] \
cl_2:[2,320,32,48] \
cl_3:[2,320,16,24] \
cl_4:[2,640,16,24] \
cl_5:[2,640,16,24] \
cl_6:[2,640,8,12] \
cl_7:[2,1280,8,12] \
cl_8:[2,1280,8,12] \
cl_9:[2,1280,4,6] \
cl_10:[2,1280,4,6] \
cl_11:[2,1280,4,6] \
cl_12:[2,1280,4,6]
