import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/code/ultralytics/runs/train/Exp_yolov11-m/weights/best.pt')  # select your model.pt path (Note: This should likely be a .pt file path for weights, not .yaml; adjust if needed)
    
    # 获取并打印 FLOPs（在预测前调用 model.info()）
    info = model.info(verbose=True)  # 这会打印模型的 params 和 GFLOPs
    flops = info[1] if isinstance(info, tuple) else None  # 提取 FLOPs（假设返回 tuple）
    print(f"Model FLOPs (GFLOPs): {flops:.2f}" if flops else "FLOPs not available")
    
    # 执行预测并获取 results
    results = model.predict(source='/root/code/ultralytics/yolo11_datasets/VisDrone2019-DET-test-dev/images',
                            imgsz=640,
                            project='runs/detect',
                            name='Exp_yolov11-m',
                            save=True,
                            # conf=0.2,
                            # iou=0.7,
                            # agnostic_nms=True,
                            # visualize=True,  # visualize model features maps
                            # line_width=2,  # line width of the bounding boxes
                            # show_conf=False,  # do not show prediction confidence
                            # show_labels=False,  # do not show prediction labels
                            # save_txt=True,  # save results as .txt file
                            # save_crop=True,  # save cropped images with results
                           )
    
    # 从 results 中计算平均推理速度和 FPS
    if results:
        total_inference_ms = sum(result.speed['inference'] for result in results)
        num_images = len(results)
        avg_inference_ms = total_inference_ms / num_images if num_images > 0 else 0
        avg_fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0
        
        print("\n--- Prediction Metrics ---")
        print(f"Average FPS (frames per second): {avg_fps:.2f}")
        print(f"Average Inference Speed (ms per image): {avg_inference_ms:.4f}")
        print("--- End Prediction Metrics ---")
    else:
        print("No results available for speed calculation.")