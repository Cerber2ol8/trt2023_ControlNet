import onnx
import argparse
import numpy as np
from onnxsim import simplify
from collections import OrderedDict
import onnx_graphsurgeon as gs

# SpatialTransformer中有两个attention块
# att1为self-attention
# att2为cross-attention

def addAttentionPlugin(sourceOnnx,destinationOnnx):
     graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))
     nlayer=0
     for node in graph.nodes:
            if node.op == 'LayerNormalization' and node.o().op=='MatMul': 
                if len(node.o().o())==3: # att1 qkv
                    #print(node.o(0).op,node.o().o().op)
                    inputs=node.inputs
                    outputs=node.o().o().o().o().outputs
                    scale=node.o().o().inputs[1]
                    AttentionN = gs.Node("Attention", "Attention-" + str(nlayer), inputs= inputs, outputs=outputs)
                    AttentionN.attrs = OrderedDict([("scale", scale)])
                    graph.nodes.append(AttentionN)
                    print("Attention-" + str(nlayer))
                    node.o().o().o().o().outputs=[]
                    nlayer +=1
     graph.cleanup()
     onnx.save(gs.export_onnx(graph), destinationOnnx)    


def cutGraph(sourceOnnx,destinationOnnx):
     graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))
     nlayer=0
     for node in graph.nodes:
            if node.op == 'LayerNormalization' and node.o().op=='MatMul': 
                if len(node.o().o())==3: # att1 qkv
                    #print(node.o(0).op,node.o().o().op)
                    inputs=node.inputs
                    outputs=node.o().o().o().o().outputs
                    scale=node.o().o().inputs[1]
                    AttentionN = gs.Node("Attention", "Attention-" + str(nlayer), inputs= inputs, outputs=outputs)
                    AttentionN.attrs = OrderedDict([("scale", scale)])
                    graph.nodes.append(AttentionN)
                    print("Attention-" + str(nlayer))
                    node.o().o().o().o().outputs=[]
                    nlayer +=1
     graph.cleanup()
     onnx.save(gs.export_onnx(graph), destinationOnnx)    

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='optimize onnx describe.')
  parser.add_argument(
      "--input_path",
      type = str,
      default="./onnxsim_model/unet.onnx",
      help="input onnx model path, default is onnxsim_model/unet.onnx")

  parser.add_argument(
      "--save_path",
      type=str,
      default="./onnxsim_model/unet.onnx",
      help="save direction of onnx models,default is onnxsim_model/unet.onnx")

  args = parser.parse_args()
  print(args)

  addAttentionPlugin(args.save_path,args.save_path)
  