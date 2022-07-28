import argparse
import logging

import tensorrt as trt

logging.basicConfig(level=logging.INFO)
logging.getLogger('build_engine').setLevel(logging.INFO)
log = logging.getLogger('build_engine')


def main(opt):

    # VERBOSE，INFO，WARNING，ERRROR，INTERNAL_ERROR
    logger = trt.Logger(trt.Logger.ERROR)
    if opt.verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    trt.init_libnvinfer_plugins(logger, namespace='')

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = opt.workspace * 1 << 30

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger)
    status = parser.parse_from_file(opt.onnx)
    if not status:
        log.error('Failed parsing .onnx file!')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    log.info(f"\nNetwork Description\n{'*'*30}")
    for i in inputs:
        print(f"Input '{i.name}' with shape {i.shape} and dtype {i.dtype}")
    for o in outputs:
        print(f"Output '{o.name}' with shape {o.shape} and dtype {o.dtype}")
    print(f"{'*'*30}")
    if opt.dynamic_batch:
        wandh = network.get_input(0).shape[-2:]
        name = network.get_input(0).name
        profile = builder.create_optimization_profile()
        log.info(f'\ndynamic batch profile is\n\
        {(opt.batch_size[0], 3, *wandh)}\n\
        {(opt.batch_size[1], 3, *wandh)}\n\
        {(opt.batch_size[2], 3, *wandh)}')
        profile.set_shape(name, (opt.batch_size[0], 3, *wandh),
                          (opt.batch_size[1], 3, *wandh),
                          (opt.batch_size[2], 3, *wandh))
        config.add_optimization_profile(profile)

    if opt.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if opt.int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = get_tools(opt)

    with builder.build_serialized_network(network, config) as engine, open(
            opt.engine, 'wb') as t:
        t.write(engine)


def get_tools(opt):
    assert opt.method in ('torch', 'cuda')
    dataloader = Calibrator = None
    if opt.method == 'torch':
        import torch
        from torch.utils.data.dataloader import DataLoader

        from utils.torch_calibrator import TorchCalibrator
        from utils.torch_datasets import TorchDataset

        dataset = TorchDataset(root=opt.calib_data)
        dataloader = DataLoader(dataset,
                                batch_size=opt.batch_size[0],
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=True,
                                collate_fn=dataset.collate_fn)
        device = torch.device(f'cuda:{opt.device}')
        Calibrator = TorchCalibrator(opt.cache, device=device)
        Calibrator.set_image_batcher(dataloader)

    elif opt.method == 'cuda':
        from utils.cuda_calibrator import CudaCalibrator
        from utils.numpy_datasets import NumpyhDataloader

        dataloader = NumpyhDataloader(root=opt.calib_data,
                                      batch_size=opt.batch_size[0])

        Calibrator = CudaCalibrator(opt.cache)
        Calibrator.set_image_batcher(dataloader)

    assert dataloader is not None and Calibrator is not None
    return Calibrator


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--onnx',
                        type=str,
                        required=True,
                        help='onnx path')
    parser.add_argument('-e',
                        '--engine',
                        type=str,
                        default=None,
                        help='engine path')
    parser.add_argument('--batch-size',
                        nargs='+',
                        type=int,
                        default=[1, 16, 32],
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

    opt.batch_size = opt.batch_size if len(
        opt.batch_size) == 3 else opt.batch_size[-1:]
    opt.batch_size.sort()
    opt.dynamic_batch = len(opt.batch_size) == 3
    opt.engine = opt.engine if opt.engine else opt.onnx.replace(
        'onnx', 'engine')
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
