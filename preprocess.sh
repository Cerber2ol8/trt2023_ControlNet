echo "preprocess"
python3 torch2onnx.py
python3 gen_trt_engine.py