# ICPR2026: 增强无人机航拍视角中的小目标检测性能
## DataSet
- Vistrone(HBB)
  - VisDrone 2019-DET-train
  - Visdrone 2019-DET-val
  - Visdrone 2019-DET-test-dev
 - COCO2017
 - DOTA-v1.5(OBB)
 - AI-TOD(HBB)
 - UAVDT(HBB)

## Baseline
YOLOv11 提供了一系列从Nano(N)到Extra Large(X)不同尺寸的模型。由于`ScaleFlow`idea的一个核心是“在给定FLOPs预算下”，这就涉及到了效率与性能的权衡，因此在不同规模的模型上进行测试是很有必要的。
- yolov11-s (small,轻量级模型)
- yolov11-M (Medium)


