from PIL import Image
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt
import numpy
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
B= 64
H = 5
W = 5
C = 1

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (8,3,224,224)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32

def add_cbr(network, input_tensor):
    
    #conv1_w = trt.Weights(numpy.random.rand(2,256,15,5).astype(np.float32))
    conv1_w = trt.Weights(numpy.random.rand(64,3,7,7).astype(np.float32))
    conv1_b = trt.Weights(np.random.rand(B,).astype(np.float32))
    #conv1_b = trt.Weights()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=B, kernel_shape=(7,7), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (2, 2)
    conv1.padding = (3,3)
    relu1 = network.add_activation(input=conv1.get_output(0), type=trt.ActivationType.RELU)
    return relu1    

def populate_network(network):
    # Configure the network layers based on the weights provided.

    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    #conv1_w = trt.Weights(numpy.random.rand(2,128,15,5).astype(np.float32)) 
    #conv1_b = trt.Weights(np.random.rand(B,).astype(np.float32))
    #conv1 = network.add_convolution(input=input_tensor, num_output_maps=B, kernel_shape=(5,5), kernel=conv1_w, bias=conv1_b)
    #conv1.stride = (1, 1)
    cbr_output0 = add_cbr(network, input_tensor)
    #relu1 = network.add_activation(input=conv1.get_output(0), type=trt.ActivationType.RELU)
    #Add a dead layer:
    cbr_output1 = add_cbr(network, input_tensor)
    #add a layer to concat
    cbr_output2 = add_cbr(network, input_tensor)
    #add a concat
    concatLayer = network.add_concatenation([cbr_output0.get_output(0),cbr_output2.get_output(0)])    
    network.mark_output(tensor=concatLayer.get_output(0))

def build_engine():
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
        builder.max_workspace_size = 1 << 20
        if(builder.platform_has_fast_fp16): 
            print("fast fp16 enabled")
            builder.fp16_mode=True
        # Populate the network with some random data and layers
        # which demonstrate layer fusion, dead layer elimination, and horizontal fusion.
        populate_network(network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


def main():
    # Do inference with TensorRT.
    engine = build_engine()
    #serialize engine
    with open('cbr_engine16', 'wb') as f:
        f.write(engine.serialize()) 

if __name__ == '__main__':
    main()
