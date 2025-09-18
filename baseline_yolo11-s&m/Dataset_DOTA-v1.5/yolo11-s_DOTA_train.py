import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

def on_train_epoch_end(trainer):
    """Callback to print custom metrics at the end of each training epoch (after validation)."""
    metrics = trainer.validator.metrics
    
    # 总体指标
    print(f"\n--- Epoch {trainer.epoch + 1} Validation Metrics ---")
    print(f"AP (mAP@0.5:0.95): {metrics.box.map:.4f}")
    print(f"AP50 (mAP@0.5): {metrics.box.map50:.4f}")
    print(f"AP75 (mAP@0.75): {metrics.box.map75:.4f}")  # 修正为 AP@0.75 以匹配常见表述
    
    # 小目标等大小分类的 AP (从 COCOeval.stats 提取，mAP50-95 for small/medium/large)
    try:
        coco_eval = trainer.validator.evaluator.coco_eval['bbox'] if 'bbox' in trainer.validator.evaluator.coco_eval else trainer.validator.evaluator.coco_eval
        ap_small = coco_eval.stats[3]  # AP_s (mAP50-95 for small objects)
        ap_medium = coco_eval.stats[4]  # AP_m
        ap_large = coco_eval.stats[5]  # AP_l
        print(f"AP_small (small objects mAP50-95): {ap_small:.4f}")
        print(f"AP_medium (medium objects mAP50-95): {ap_medium:.4f}")
        print(f"AP_large (large objects mAP50-95): {ap_large:.4f}")
    except (AttributeError, KeyError):
        # Fallback if coco_eval not accessible or not initialized (common for non-COCO OBB datasets like DOTA)
        print("AP_small/AP_medium/AP_large: Not available (COCO eval not initialized for OBB dataset)")
    
    # FPS (推理速度, ms)
    inference_ms = metrics.speed['inference']
    fps = 1000 / inference_ms if inference_ms > 0 else 0  # 计算 FPS
    print(f"FPS (frames per second): {fps:.2f}")
    print(f"Inference Speed (ms per image): {inference_ms:.4f}")
    
    print("--- End Epoch Metrics ---\n")

if __name__ == '__main__':
    model = YOLO("yolo11s-obb.pt")  # 载入预训练权重
    
    # 打印 FLOPs (在训练开始前) - 修正提取逻辑 (info 返回 dict)
    info = model.info(verbose=True)  # 返回 dict 如 {'params': ..., 'flops': ...}
    flops = info.get('flops', 0) / 1e9  # 转换为 GFLOPs
    print(f"Model FLOPs (GFLOPs): {flops:.2f}")
    
    # 添加自定义回调
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    # 训练（替换为你的 YAML 路径）
    results = model.train(
        data='/root/autodl-tmp/DOTA15_yolo_obb/dota15-obb.yaml',  # 改为你的 dota15-obb.yaml 路径
        epochs=200,  # 可调整
        imgsz=1024,  # 常见于 DOTA，基于文档建议
        batch=8,
        device=0,
        workers=8,
        optimizer='SGD',
        project='runs/train',
        name='exp-yolo11s-obb',
        save_json=True,  # 保存详细 JSON，用于潜在分析 (但注意数据集前缀问题)
        patience=50,  # 添加 early stopping 以优化训练
    )
    
    # 训练后最终验证（修正：设置 save_json=False 以避免 IndexError）
    metrics = model.val(save_json=False)  # 避免 JSON 解析错误；如果需要 JSON，修复数据集前缀
    print("Final Training Complete. Last validation metrics printed above.")
    print("Loss plots: Check runs/train/exp-yolo11s-obb/plots/loss.png and results.png")