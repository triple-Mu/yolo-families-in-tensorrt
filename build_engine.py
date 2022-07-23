import argparse

import tensorrt as trt


def main_torch(opt):
    import torch
    from torch.utils.data.dataloader import DataLoader

    from utils.torch_calibrator import TorchCalibrator
    from utils.torch_datasets import TorchDataset

    dataset = TorchDataset(root=opt.calib_data)
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=dataset.collate_fn)

    device = torch.device(f'cuda:{opt.device}')
    logger = trt.Logger(
        trt.Logger.INFO)  # VERBOSE，INFO，WARNING，ERRROR，INTERNAL_ERROR
    if opt.verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    trt.init_libnvinfer_plugins(logger, namespace='')

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = opt.workspace * 1 << 30

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(opt.onnx)

    if opt.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if opt.int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)

    # calib_shape = [opt.batch_size, 3, *opt.imgsz]
    # calib_dtype = trt.nptype(trt.float32)
    Calibrator = TorchCalibrator(opt.cache, device=device)
    Calibrator.set_image_batcher(dataloader)
    config.int8_calibrator = Calibrator

    with builder.build_serialized_network(network, config) as engine, open(
            opt.engine, 'wb') as t:
        t.write(engine)


def main_cuda(opt):
    from utils.cuda_calibrator import CudaCalibrator
    from utils.numpy_datasets import NumpyhDataloader

    dataloader = NumpyhDataloader(root=opt.calib_data,
                                  batch_size=opt.batch_size)

    logger = trt.Logger(
        trt.Logger.INFO)  # VERBOSE，INFO，WARNING，ERRROR，INTERNAL_ERROR
    if opt.verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    trt.init_libnvinfer_plugins(logger, namespace='')

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = opt.workspace * 1 << 30

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(opt.onnx)

    if opt.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if opt.int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)

    # calib_shape = [opt.batch_size, 3, *opt.imgsz]
    # calib_dtype = trt.nptype(trt.float32)
    Calibrator = CudaCalibrator(opt.cache)
    Calibrator.set_image_batcher(dataloader)
    config.int8_calibrator = Calibrator

    with builder.build_serialized_network(network, config) as engine, open(
            opt.engine, 'wb') as t:
        t.write(engine)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='onnx path')
    parser.add_argument('--engine', type=str, default=None, help='engine path')
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='batch_size of tensorrt engine')
    parser.add_argument(
        '--imgsz',
        '--img',
        '--img-size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='image (h, w)',
    )

    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='cuda device, i.e. 0 or 0,1,2,3',
    )
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print verbose log')
    parser.add_argument('--workspace',
                        type=int,
                        default=8,
                        help='max workspace GB')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='build fp16 network')
    parser.add_argument('--int8',
                        action='store_true',
                        help='build int8 network')
    parser.add_argument('--calib-data',
                        type=str,
                        default='./calib_data',
                        help='calib data for int8 calibration')
    parser.add_argument('--cache',
                        type=str,
                        default='./calib.cache',
                        help='calib cache for int8 calibration')
    parser.add_argument('--method',
                        type=str,
                        default='torch',
                        help='calib dataloader, you can choose torch or cuda')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    if opt.method == 'torch':
        main_torch(opt)
    elif opt.method == 'cuda':
        main_cuda(opt)
