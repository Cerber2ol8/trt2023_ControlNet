# 天池 NVIDIA TensorRT Hackathon 2023 —— 生成式AI模型优化赛初赛

针对StableDiffusion模型canny pipeline优化
包含的处理流程： 拆分为多个onnx，转换tensorRT引擎, 自定义算子插件加载

涉及到的优化处理:

图像生成pipeline优化

muti-stream和cuda stream

trt fp16量化

trt插件算子融合，LayerNormPlugin（抄的）
