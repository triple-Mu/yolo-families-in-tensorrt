import tensorrt as trt
import numpy as np
import torch
import ctypes
import time
from collections import OrderedDict,namedtuple



def run0(img):
    device = torch.device("cuda:0")
    engine = "trtexec/yolov5s.engine"
    imgsz = [1, 3, 640, 640]
    gpu_tensor = torch.cuda.FloatTensor(*imgsz)
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(str(engine), 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        print(shape)
        data = torch.from_numpy(np.empty(shape,
                                         dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data,
                                 int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()
    # WarmUp
    tmps = [torch.randn(imgsz).to(device) for _ in range(10)]
    for i in range(10):
        binding_addrs['images'] = int(tmps[i].data_ptr())
        context.execute_v2(list(binding_addrs.values()))
    # Infer
    gpu_tensor.data[...] = img.
    binding_addrs['images'] = int(random.choice(gpuTensors).data_ptr())
    context.execute_v2(list(binding_addrs.values()))

if __name__ == '__main__':
    img = np.load("image.npy")


