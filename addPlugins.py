
import onnx
import argparse
import numpy as np
from onnxsim import simplify
from collections import OrderedDict
import onnx_graphsurgeon as gs
import os 

def addLayerNormPlugin(sourceOnnx,destinationOnnx):
    bLayerNormPlugin = True
    nLayerNormPlugin = 0
    graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

    if bLayerNormPlugin:
        for node in graph.nodes:
            if node.op == 'LayerNormalization':
                inputTensor = node.inputs[0]
                weight = node.inputs[1]
                bias = node.inputs[2]

                layerNormNOut = gs.Variable("layerNormNOut-" + str(nLayerNormPlugin),dtype=np.float32,shape=inputTensor.shape)

                layerNormN = gs.Node("LayerNorm", "LayerNorm-" + str(nLayerNormPlugin), inputs=[inputTensor],outputs=[layerNormNOut])
                

                CustomLinear = gs.Node("CustomLinear", "CustomLinear-" + str(nLayerNormPlugin), inputs=[layerNormN.outputs[0]], outputs=node.outputs)
                CustomLinear.attrs = OrderedDict([("weight", weight),("bias", bias)]) 
                
                graph.nodes.append(layerNormN)
                graph.nodes.append(CustomLinear)

                print("LayerNorm-" + str(nLayerNormPlugin))
                nLayerNormPlugin += 1
                node.outputs = []
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), destinationOnnx)

def addAttentionPlugin(sourceOnnx,destinationOnnx):
     graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))
     nlayer=0
     for node in graph.nodes:
            if node.op == 'Split' and node.o().op=='MatMul' and node.o().o().op=='Mul':#
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

def findAttention(sourceOnnx):
    graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))
    nlayer=0
    self_attn = []
    for node in graph.nodes:
        if node.op == 'CustomLinear' and node.o().op=='Cast': #self-attn
                
                if len(node.o().outpus) == 3:
                     
                     self_attn.append(node.name)

                

                #print(node)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='optimize onnx describe.')
    parser.add_argument(
        "--input_path",
        type = str,
        default='./onnxsim_model/unet.onnx')

    parser.add_argument(
        "--save_path",
        type=str,
        default='./onnxsim_model/unet.onnx')

    args = parser.parse_args()

    #addLayerNormPlugin(args.input_path, args.save_path)
    


  