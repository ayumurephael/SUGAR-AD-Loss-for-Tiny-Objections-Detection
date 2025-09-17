# val.py
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/code/ultralytics/runs/train/exp-yolov11m-obb/weights/best.pt') #修改为yolo11-m-obb训练后的最好模型位置
    
    # 获取并打印 FLOPs（在验证前调用 model.info()）
    info = model.info(verbose=True)  # 这会打印模型的 params 和 GFLOPs
    flops = info[1] if isinstance(info, tuple) else None  # 提取 FLOPs（假设返回 tuple）
    print(f"Model FLOPs (GFLOPs): {flops:.2f}" if flops else "FLOPs not available")
    
    # 执行验证并获取 metrics
    metrics = model.val(
        data='/root/autodl-tmp/DOTA15_yolo_obb/dota15-obb.yaml',
        split='val',
        imgsz=1024,  # 与训练时一致
        batch=16,
        device=0,
        workers=8,
        save_json=True,  # 如果需要计算详细指标
        project='runs/val',
        name='exp-yolov11m-obb',
    )
    
    # 从 metrics 中提取并打印所需指标
    print("\n--- Validation Metrics ---")
    print(f"AP (mAP@0.5:0.95): {metrics.box.map:.4f}")
    print(f"AP50 (mAP@0.5): {metrics.box.map50:.4f}")
    print(f"AP75 (mAP@0.75): {metrics.box.map75:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")  # 与 AP 相同
    
    # 小目标等大小分类的 AP (从 COCOeval.stats 提取，mAP50-95 for small/medium/large)
    try:
        coco_eval = metrics.evaluator.coco_eval['bbox'] if 'bbox' in metrics.evaluator.coco_eval else metrics.evaluator.coco_eval
        ap_small = coco_eval.stats[3]  # AP_s (mAP50-95 for small objects)
        ap_medium = coco_eval.stats[4]  # AP_m
        ap_large = coco_eval.stats[5]  # AP_l
        print(f"AP_small (small objects mAP50-95): {ap_small:.4f}")
        print(f"AP_medium (medium objects mAP50-95): {ap_medium:.4f}")
        print(f"AP_large (large objects mAP50-95): {ap_large:.4f}")
    except (AttributeError, KeyError):
        # Fallback if coco_eval not accessible or not initialized (common for non-COCO OBB datasets like DOTA)
        print("AP_small/AP_medium/AP_large: Not available (COCO eval not initialized for OBB dataset)")
    
    # 计算并打印 FPS 和推理速度
    inference_ms = metrics.speed['inference']
    fps = 1000 / inference_ms if inference_ms > 0 else 0
    print(f"FPS (frames per second): {fps:.2f}")
    print(f"Inference Speed (ms per image): {inference_ms:.4f}")
    print("--- End Validation Metrics ---")