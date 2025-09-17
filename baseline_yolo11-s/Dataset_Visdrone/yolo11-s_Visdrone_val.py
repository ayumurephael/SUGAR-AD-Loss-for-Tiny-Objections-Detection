import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('code/ultralytics/runs/train/exp3/weights/best.pt')
    
    # 获取并打印 FLOPs（在验证前调用 model.info()）
    info = model.info(verbose=True)  # 这会打印模型的 params 和 GFLOPs
    flops = info[1] if isinstance(info, tuple) else None  # 提取 FLOPs（假设返回 tuple）
    print(f"Model FLOPs (GFLOPs): {flops:.2f}" if flops else "FLOPs not available")
    
    # 执行验证并获取 metrics
    metrics = model.val(data='/root/code/ultralytics/ultralytics/cfg/datasets/VisDrone.yaml',
                        split='val',
                        imgsz=640,
                        batch=16,
                        # iou=0.7,
                        # rect=False,
                        # save_json=True,  # 如果需要计算 COCO 指标（如 AP_small 等），可以启用
                        project='runs/val',
                        name='exp',
                       )
    
    # 从 metrics 中提取并打印所需指标
    print("\n--- Validation Metrics ---")
    print(f"AP (mAP@0.5:0.95): {metrics.box.map:.4f}")  # AP 假设为 mAP@0.5:0.95
    print(f"AP50 (mAP@0.5): {metrics.box.map50:.4f}")
    print(f"AP75 (mAP@0.75): {metrics.box.map75:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")  # 与 AP 相同，如果有重复
    
    # 计算并打印 FPS 和推理速度
    inference_ms = metrics.speed['inference']
    fps = 1000 / inference_ms if inference_ms > 0 else 0
    print(f"FPS (frames per second): {fps:.2f}")
    print(f"Inference Speed (ms per image): {inference_ms:.4f}")
    print("--- End Validation Metrics ---")