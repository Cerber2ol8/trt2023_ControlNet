
echo "preprocess"

cd ./plugin/CustomLinearPlugin
make all
mv CustomLinear.so ../target/CustomLinear.so
cd ..

cd ./plugin/LayerNormPlugin
make all
mv LayerNorm.so ../target/LayerNorm.so
cd ..



python3 torch2onnx.py

mkdir onnxsim_model

onnxsim onnx_models/control_net.onnx onnxsim_model/control_net.onnx
onnxsim onnx_models/vae_decoder.onnx onnxsim_model/vae_decoder.onnx
onnxsim onnx_models/unet.onnx onnxsim_model/unet.onnx
onnxsim onnx_models/FrozenCLIPEmbedder.onnx onnxsim_model/FrozenCLIPEmbedder.onnx


python3 addPlugins.py --input_path ./onnxsim_model/unet.onnx --save_path ./onnxsim_model/unet.onnx
# python3 addPlugins.py --input_path ./onnxsim_model/control_net.onnx --save_path ./onnxsim_model/control_net.onnx



python3 convert2trt.py --model_name unet
python3 convert2trt.py --model_name control_net
