# detect.py
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/code/ultralytics/runs/train/exp-yolov11m-obb/weights/best.pt')
    
    # 获取并打印 FLOPs（在预测前调用 model.info()）
    info = model.info(verbose=True)  # 这会打印模型的 params 和 GFLOPs
    flops = info[1] if isinstance(info, tuple) else None  # 提取 FLOPs（假设返回 tuple）
    print(f"Model FLOPs (GFLOPs): {flops:.2f}" if flops else "FLOPs not available")
    
    # 执行预测并获取 results
    results = model.predict(
        source='/root/autodl-tmp/DOTA15_yolo_obb/images/test',
        imgsz=1024,  # 与训练时一致
        batch=16,
        device=0,
        workers=8,
        project='runs/detect',
        name='exp-yolov11m-obb',
        save=True,
    )
    
    # 从 results 中计算平均推理速度和 FPS
    # 注意：对于 detect/predict，无法计算 AP 等验证指标，因为缺少 ground truth 标签
    if results:
        total_inference_ms = sum(result.speed['inference'] for result in results)
        num_images = len(results)
        avg_inference_ms = total_inference_ms / num_images if num_images > 0 else 0
        avg_fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0
        
        print("\n--- Prediction Metrics ---")
        print(f"Average FPS (frames per second): {avg_fps:.2f}")
        print(f"Average Inference Speed (ms per image): {avg_inference_ms:.4f}")
        print("AP/AP50/AP75/mAP@0.5:0.95/AP_small/AP_medium/AP_large: Not available (no ground truth for validation)")
        print("--- End Prediction Metrics ---")
    else:
        print("No results available for speed calculation.")