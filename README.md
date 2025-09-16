# ICPR2026: 增强无人机航拍视角中的小目标检测性能
Chun-chieh Han , 2025 ,BIT

## DataSet
- Vistrone
  - VisDrone 2019-DET-train
  - Visdrone 2019-DET-val
  - Visdrone 2019-DET-test-dev
 - AI-TOD
 - DOTA-v1.5

## Baseline
YOLOv11 提供了一系列从Nano(N)到Extra Large(X)不同尺寸的模型。由于`ScaleFlow`idea的一个核心是“在给定FLOPs预算下”，这就涉及到了效率与性能的权衡，因此在不同规模的模型上进行测试是很有必要的。
- yolov11-s (small,轻量级模型)
- yolov11-M (Medium)

基线实验：

量化yolov11-s和yolov11-M/B在**没有任何修改**的情况下，在VisDrone/MS COCO/AI-TOD数据集上的各项检测指标

yolov11-s漏检可能的原因：
- 目标被环境淹没了(环境噪声、环境与目标太像了) --> Solutions:原图看一次，去掉能去掉的背景-->Zoom in 放大再看一次
- 可以先考虑预处理时进行清晰度改善、图像增强
- 图像清晰度调高-->将figure分为若干小块/Zoom in & out-->yolo检测器

不同尺度物体的检测
- 幅度感知线性注意力模块,Magnitude-Aware Linear Attention,MALA-->增强对不同目标的局部和全局感知，根据目标的大小和类别动态调整
- 多感受野聚合卷积模块,Multi-Receptive Field Aggregation,MRFA-->单一感受野检测容易误检，通过MFRA多感受野聚合-->**漏检**(yolo内部改进)
- 通道与空间注意力模块,Channel and Spatial Attention Block,CASAB-->背景信息冗余度大、小目标容易被背景淹没
- 离散余弦变换的高频感知模块,High Frequency Perception Moduel,HFP-->图像预处理
  - 高频特征提取：使用高通滤波器对输入特征图处理，过滤低频背景成分，保留小目标边缘、纹理等高频细节
  - 双通道注意力增强，突出小目标的特征通道、聚焦小目标所在区域
- 傅里叶频率特征网络,FreMLP-->提升图像清晰度

YOLO网络内部进行修改：
- 特征互补映射模块,Feature Complementary Mapping Module,FCM-->在特征金字塔网络FPN中嵌入FCM，实现多尺度目标检测
- 多核感知卷积单元,Multi-Kernel Perception Unit,MKP-->尺度跨度大、复杂背景导致特征模糊
- 多尺度二值化分组扩张卷积,Multi-scale grouped dilated convolution,MSGDC-->在特征提取阶段，引入MSGDC，不同扩张率的卷积核可以获得不同感受野的特征，小扩张率卷积核关注细节，用于检测小目标；大扩张率卷积核有助于识别大目标





