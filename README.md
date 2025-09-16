# ICPR2026: 增强无人机航拍视角中的小目标检测性能
Chun-chieh Han , 2025 ,BIT

## DataSet
- Vistrone
  - VisDrone 2019-DET-train
  - Visdrone 2019-DET-val
  - Visdrone 2019-DET-test-dev
 - AI-TOD

## Baseline
YOLOv11 提供了一系列从Nano(N)到Extra Large(X)不同尺寸的模型。由于`ScaleFlow`idea的一个核心是“在给定FLOPs预算下”，这就涉及到了效率与性能的权衡，因此在不同规模的模型上进行测试是很有必要的。
- yolov11-s (small,轻量级模型)
- yolov11-M (Medium)

基线实验：

量化yolov11-s和yolov11-M/B在**没有任何修改**的情况下，在VisDrone/MS COCO/AI-TOD数据集上的各项检测指标

yolov11-s漏检可能的原因：
- 目标被环境淹没了(环境噪声、环境与目标太像了) --> Solutions:原图看一次，去掉能去掉的背景-->Zoom in 放大再看一次
- 可以先考虑预处理时进行清晰度改善、图像增强


