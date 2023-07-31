
echo "preprocess"
python3 torch2onnx.py

mkdir onnx_models_opti

python3 -m onnxoptimizer onnx_models/control_net.onnx onnx_models_opti/control_net.onnx
python3 -m onnxoptimizer onnx_models/vae_decoder.onnx onnx_models_opti/vae_decoder.onnx
python3 -m onnxoptimizer onnx_models/unet.onnx onnx_models_opti/unet.onnx
python3 -m onnxoptimizer onnx_models/FrozenCLIPEmbedder.onnx onnx_models_opti/FrozenCLIPEmbedder.onnx

python3 gen_trt_engine.py
