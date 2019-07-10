import sys
import tensorrt as trt
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit

DTYPE = trt.float32

def allocate_buffers(engine):
    print('allocate buffers')

    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(DTYPE))
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    return h_input, d_input, h_output, d_output


def do_inference(context, h_input, d_input, h_output, d_output):
    # Transfer input data to the GPU.
    cuda.memcpy_htod(d_input, h_input)

    # Run inference.
    st = time.time()
    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
    print('Inference time: {} [msec]'.format((time.time() - st)*1000))

    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh(h_output, d_output)

    return h_output

def main():
  engine_file = sys.argv[1]
  engine = None
  TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
  with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

  h_input, d_input, h_output, d_output = allocate_buffers(engine)
  #h_input = np.random.rand(1,40,40)
  with engine.create_execution_context() as context:
    for i in range(10):
      output = do_inference(context, h_input, d_input, h_output, d_output)



if __name__ == "__main__":
   main()
