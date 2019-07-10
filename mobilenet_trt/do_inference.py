import sys
import tensorrt as trt
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image

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

INPUT_SHAPE = (3, 224, 224)
def load_input(img_path, host_buffer):
    print('load input')

    with Image.open(img_path) as img:
        c, h, w = INPUT_SHAPE
        dtype = trt.nptype(DTYPE)
        img_array = np.asarray(img.resize((w, h), Image.BILINEAR)).transpose([2, 0, 1]).astype(dtype).ravel()
        img_array = img_array / 127.5 - 1.0

    np.copyto(host_buffer, img_array)

def main():
  engine_file = sys.argv[1]
  img_file = sys.argv[2]

  engine = None
  TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
  with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
  LABELS = "mobilenet_labels.txt"
  with open(LABELS) as f:
     labels = f.read().split('\n')
  h_input, d_input, h_output, d_output = allocate_buffers(engine)
  #h_input = np.random.rand(1,40,40)
  load_input(img_file, h_input)
  with engine.create_execution_context() as context:
    for i in range(1):
      output = do_inference(context, h_input, d_input, h_output, d_output)
      pred_idx = np.argsort(output)[::-1]
      pred_prob = np.sort(output)[::-1]

      print('\n%s is a ' % (img_file))
      for i in range(6):
          print('{} {} {:.5f}'.format(i + 1, labels[pred_idx[i]], pred_prob[i]))



if __name__ == "__main__":
   main()
