import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

def on_train_epoch_end(trainer):
    """Callback to print custom metrics at the end of each training epoch (after validation)."""
    metrics = trainer.validator.metrics
    
    # 总体指标
    print(f"\n--- Epoch {trainer.epoch + 1} Validation Metrics ---")
    print(f"AP (mAP@0.5:0.95): {metrics.box.map:.4f}")
    print(f"AP50 (mAP50): {metrics.box.map50:.4f}")
    print(f"AP75 (mAP75): {metrics.box.map75:.4f}")
    
    # 小目标等大小分类的 AP (从 COCOeval.stats 提取，mAP50-95 for small/medium/large)
    try:
        coco_eval = trainer.validator.evaluator.coco_eval['bbox'] if 'bbox' in trainer.validator.evaluator.coco_eval else trainer.validator.evaluator.coco_eval
        ap_small = coco_eval.stats[3]  # AP_s (mAP50-95 for small objects)
        ap_medium = coco_eval.stats[4]  # AP_m
        ap_large = coco_eval.stats[5]   # AP_l
        print(f"AP_small (small objects mAP50-95): {ap_small:.4f}")
        print(f"AP_medium (medium objects mAP50-95): {ap_medium:.4f}")
        print(f"AP_large (large objects mAP50-95): {ap_large:.4f}")
    except AttributeError:
        # Fallback if coco_eval not accessible
        print("AP_small/AP_medium/AP_large: Not available (COCO eval not initialized)")
    
    # Per-class AP50 和 mAP50-95（数组，索引对应类别）
    print("Per-class AP50:", [f"{x:.4f}" for x in metrics.box.ap50])
    print("Per-class mAP50-95:", [f"{x:.4f}" for x in metrics.box.ap])
    
    # 自定义：APt (假设为 tricycle, class 6) 和 APvt (车辆平均：car=3, van=4, truck=5, bus=8, motor=9)
    if len(metrics.box.ap50) > 6:
        ap_t = metrics.box.ap50[6]  # AP50 for tricycle
        vehicle_classes = [3, 4, 5, 8, 9]
        ap_vt = sum(metrics.box.ap50[i] for i in vehicle_classes if i < len(metrics.box.ap50)) / len(vehicle_classes)
        print(f"APt (tricycle AP50): {ap_t:.4f}")
        print(f"APvt (vehicles avg AP50): {ap_vt:.4f}")
    else:
        print("Custom APt/APvt: Insufficient classes")
    
    # FPS (推理速度, ms)
    inference_ms = metrics.speed['inference']
    fps = 1000 / inference_ms if inference_ms > 0 else 0  # 计算 FPS
    print(f"FPS (frames per second): {fps:.2f}")
    print(f"Inference Speed (ms per image): {inference_ms:.4f}")
    
    print("--- End Epoch Metrics ---\n")

if __name__ == '__main__':
    model = YOLO('/root/code/ultralytics/ultralytics/cfg/models/11/yolo11.yaml') 
    model.load('yolo11s.pt')  # 载入预训练权重
    
    # 打印 FLOPs (在训练开始前)
    info = model.info(verbose=True)  # 这会打印 params 和 GFLOPs
    flops = info[1] if isinstance(info, tuple) else None  # 假设 info 返回 (params, flops)
    print(f"Model FLOPs (GFLOPs): {flops:.2f}" if flops else "FLOPs not available")
    
    # 添加自定义回调
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    # 训练（替换为你的 VisDrone YAML 路径）
    results = model.train(
        data='/root/code/ultralytics/ultralytics/cfg/datasets/VisDrone.yaml',  # 改为 VisDrone.yaml
        cache=False,
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        close_mosaic=0,
        workers=8,
        optimizer='SGD',
        project='runs/train',
        name='exp',
        save_json=True,  # 保存详细 COCO JSON，用于分析小目标等
        # patience=0,  # 如需关闭 early stop
        # resume=True,  # 如需断点续训
    )
    
    # 训练后最终验证（可选，已在回调中覆盖）
    metrics = model.val()
    print("Final Training Complete. Last validation metrics printed above.")