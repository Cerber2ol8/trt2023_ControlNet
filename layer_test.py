
import onnx
import argparse
import numpy as np
from onnxsim import simplify
from collections import OrderedDict
import onnx_graphsurgeon as gs

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
                layerNormN = gs.Node("LayerNorm", "LayerNorm-" + str(nLayerNormPlugin), inputs=[inputTensor], outputs=node.outputs)
                layerNormN.attrs = OrderedDict([("weight", weight),("bias", bias)]) 
                graph.nodes.append(layerNormN)
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
    for node in graph.nodes:
        if node.op == 'Cast' and node.o().op=='MatMul':#
                

                nChildNodes = len(node.outputs[0].outputs)

                if(nChildNodes == 1): # attn2
                     #print("attn1",node.op)
                     
                     pass
                if(nChildNodes == 3): # attn1
                     print("attn2",node.op)
                     
                     
                     
                     for i in range(nChildNodes):
                          mut_node = node.outputs[0].outputs[i]
                          inputs = mut_node.inputs

                          #inputs = node.o().inputs
                          print(mut_node.name)

                          pass
                     
                     


                #print(node)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='optimize onnx describe.')
    parser.add_argument(
        "--input_path",
        type = str,
        default="./target/MobileViT.onnx",
        help="input onnx model path, default is ./target/MobileViT.onnx.")

    parser.add_argument(
        "--save_path",
        type=str,
        default="./target/MobileViT_final.onnx",
        help="save direction of onnx models,default is ./target.")

    args = parser.parse_args()
    #print(args)
    args.input_path = './onnxsim_model/unet.onnx'
    args.save_path = './onnxsim_model/unet_l.onnx'
    addLayerNormPlugin(args.input_path, args.save_path)


  #findAttention(args.input_path)
  
#   addLayerNormPlugin(args.input_path,args.save_path)
#   addAttentionPlugin(args.save_path,args.save_path)
  