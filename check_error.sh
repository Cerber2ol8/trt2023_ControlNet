polygraphy run onnxsim_model/FrozenCLIPEmbedder.onnx \
    --trt --onnxrt \
    --trt-outputs mark all \
    -- onnx-outputs mark all \
    --atol 1e-2 --rtol 1e-3 \
    --fail-fast 