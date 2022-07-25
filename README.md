# YOLO-TensorRT8

```YOLO-TensorRT8```  是基于 [```TensorRT8```](https://developer.nvidia.com/nvidia-tensorrt-8x-download)  和  [```Efficient NMS Plugin```](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin)  实现的端到端检测架构。

利用  ```PyTorch```  中的 `torch.autograd.Function` 和 `symbolic` 实现了 `Efficient NMS Plugin` 注册到 onnx 中。

### 展示效果

效果图如图所示:

![image](https://user-images.githubusercontent.com/92794867/179765688-2d6fd843-4440-4591-b04f-e804eff2ee7f.png)

### 输出含义和类型

图中的输出共计 4 个，分别表示:

- `num_dets` : Batch 中每张图片检测的目标数量, 类型为 `int32`;
- `det_boxes` : Batch 中每张图片中每个目标的检测框, 目标框格式为左上角和右下角坐标(x0,y0,x1,y1), 类型为 `float32`;
- `det_scores` : Batch 中每张图片中每个目标的置信度, 类型为 `float32`;
- `det_classes` : Batch 中每张图片中每个目标的类别 `id`,类型为 `int32`;


### 使用方法

``` shell
# 安装环境
pip install -r requirements.txt

# 导出普通的 ONNX
python ./export.py \
    --weights ./yolov7.pt \
    --version 7 \
    --img-size 640 \
    --batch-size 1 \
    --device cpu \
    --simplify \
    --opset 12

# 导出带有 Efficient NMS 的 ONNX
python ./export.py \
    --weights ./yolov7.pt \
    --version 7 \
    --img-size 640 \
    --batch-size 1 \
    --device cpu \
    --simplify \
    --opset 12 \
    --end2end \
    --max-obj 100 \
    --iou-thres 0.65 \
    --score-thres 0.35

# 转换 ONNX 为 TensorRT engine
trtexec --onnx=./yolov7.onnx \
	--saveEngine=./yolov7.engine \
	--fp16 # 如果使用 fp16
```

### 参数介绍

- `--weights` : 使用最新版本开源仓库训练好的模型权重。
- `--version` : 指定模型版本。目前支持 5:yolov5, 6:yolov6, 7:yolov7; 未来考虑增加 8:airdet, -1:ppyoloe(需要安装PaddleDetection)。
- `--img-size` : 输入图片尺寸。
- `--batch-size` : 输入批量大小。
- `--device` : 指定使用 CPU 或者 CUDA:0 CUDA:1 ...导出 ONNX。
- `--simplify` : 使用 onnx-simplifier 简化模型。需要安装最新版本的 onnx-simplifier。
- `--opset` : 导出支持 opset 版本算子的 ONNX。
- `--end2end` : 导出端到端的带有 Efficient NMS 的 ONNX。
- `--max-obj` : 每张图片的最大检测框数量。
- `--iou-thres` : NMS 算法中的 IOU 阈值。
- `--conf-thres` : NMS 算法中的置信度阈值。

### TensorRT-INT8 量化

准备模型训练时使用的图片放到 `./calib_data` 下，路径格式如下：

``` shell
# 文件名随意，文件后缀最好是 .jpg
./calib_data/0001.jpg
./calib_data/0002.jpg
./calib_data/0003.jpg
./calib_data/0004.jpg
./calib_data/0005.jpg
....
```

本仓库支持 `pytorch` 和 `cuda-python` 作为量化后端喂给 TensorRT 校准数据。

用法：

``` shell
# pytorch + TensorRT int8量化
python ./build_engine.py \
	--onnx yolov7.onnx \
	--engine yolov7.engine \
	--batch-size 1 \ # 需要和前文导出 onnx 时指定一致
	--imgsz 640 \
	--device 0 \
	--verbose \ # 打印详细日志(未来会使用它增加绘制 TensorRT engine 结构的功能)
	--workspace 8 \
	--fp16 \ # 如果需要 fp16 + int8 可以同时开启
	--int8 \
	--calib-data ./calib_data \ # 校准图片路径可以自定义
	--cache ./calib.cache \ # 校准集缓存,方便下次使用
	--method torch
```
