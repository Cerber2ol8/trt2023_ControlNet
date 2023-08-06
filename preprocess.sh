
echo "preprocess"
python3 torch2onnx.py

mkdir onnxsim_model

onnxsim onnx_models/control_net.onnx onnxsim_model/control_net.onnx
onnxsim onnx_models/vae_decoder.onnx onnxsim_model/vae_decoder.onnx
onnxsim onnx_models/unet.onnx onnxsim_model/unet.onnx
onnxsim onnx_models/FrozenCLIPEmbedder.onnx onnxsim_model/FrozenCLIPEmbedder.onnx

python3 gen_trt_engine.py
