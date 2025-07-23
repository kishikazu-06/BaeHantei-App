import sys
import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import requests

# --- ランタイムとPylanceの両方でyolov5を解決するための設定 ---
# 1. ランタイムのために、yolov5ディレクトリへの絶対パスをシステムパスに追加
#    app.pyファイル自身の場所を基準にするため、どこから実行してもパスがずれない
APP_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOV5_PATH = os.path.join(APP_DIR, 'yolov5')
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

from yolov5.models.yolo import Model
from yolov5.models.common import AutoShape

# PyTorch 2.6+ のセキュリティ変更に対応
try:
    torch.serialization.add_safe_globals([AutoShape, Model])
except (ImportError, AttributeError):
    pass

# 2. Pylanceのために、.vscode/settings.json に "python.analysis.extraPaths": ["./yolov5"] を設定済み

def load_model():
    model_config_path = os.path.abspath(os.path.join(YOLOV5_PATH, 'models', 'yolov5s.yaml'))
    model_weights_path = os.path.abspath(os.path.join(YOLOV5_PATH, 'yolov5s.pt'))

    # yolov5s.yamlのダウンロード
    if not os.path.exists(model_config_path):
        print(f"Downloading yolov5s.yaml to {model_config_path}")
        os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/ultralytics/yolov5/v6.1/models/yolov5s.yaml"
        r = requests.get(url, allow_redirects=True)
        with open(model_config_path, 'wb') as f:
            f.write(r.content)
        print("yolov5s.yaml downloaded.")

    # yolov5s.ptのダウンロード (app.pyからダウンロードロジックを削除したため、ここでは存在チェックのみ)
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found at {model_weights_path}. Please ensure it's in the repository.")
        raise FileNotFoundError(f"Model weights file not found at {model_weights_path}")

    print(f"--- Debugging Model Loading ---")
    print(f"APP_DIR: {APP_DIR}")
    print(f"YOLOV5_PATH: {YOLOV5_PATH}")
    print(f"model_config_path: {model_config_path}")
    print(f"model_weights_path: {model_weights_path}")
    print(f"Does model_config_path exist? {os.path.exists(model_config_path)}")
    print(f"Does model_weights_path exist? {os.path.exists(model_weights_path)}")

    # yolov5ディレクトリの内容をリストアップ (デバッグ用)
    if os.path.exists(YOLOV5_PATH) and os.path.isdir(YOLOV5_PATH):
        print(f"Contents of {YOLOV5_PATH}: {os.listdir(YOLOV5_PATH)}")
    else:
        print(f"{YOLOV5_PATH} does not exist or is not a directory.")

    # yolov5/models ディレクトリの内容をリストアップ (デバッグ用)
    models_dir = os.path.join(YOLOV5_PATH, 'models')
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        print(f"Contents of {models_dir}: {os.listdir(models_dir)}")
    else:
        print(f"{models_dir} does not exist or is not a directory.")
    print(f"--- End Debugging Model Loading ---")

    if not os.path.exists(model_config_path):
        print(f"Error: Model config file not found at {model_config_path}")
        raise FileNotFoundError(f"Model config file not found at {model_config_path}")
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found at {model_weights_path}")
        raise FileNotFoundError(f"Model weights file not found at {model_weights_path}")

    try:
        # モデルのインスタンス化 (yolov5s.yamlからアーキテクチャを定義)
        model = Model(cfg=model_config_path)

        # state_dictのロード (yolov5s.ptから重みをロード)
        state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict, strict=False)

        # AutoShapeでラップ
        if not isinstance(model, AutoShape):
            model = AutoShape(model)
        model.conf = 0.05 # 信頼度閾値を調整
        model.iou = 0.01 # NMSのIoU閾値を調整 (デバッグ用)

        # COCOデータセットのクラス名を設定
        model.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        print("Model loaded successfully using state_dict.")
        print(f"Type of loaded model: {type(model)}")
        if hasattr(model, 'names'):
            print(f"Model class names: {model.names}")
        else:
            print("Model has no 'names' attribute.")
    except Exception as e:
        print(f"Error loading model with state_dict: {e}")
        raise
    
    return model

if __name__ == "__main__":
    print("Attempting to load model and perform detection from debug_detection.py...")
    try:
        model = load_model()
        print("Model loaded successfully in debug script.")

        # サンプル画像をロード
        sample_image_path = os.path.join(YOLOV5_PATH, 'data', 'images', 'zidane.jpg')
        if not os.path.exists(sample_image_path):
            print(f"Error: Sample image not found at {sample_image_path}. Please ensure yolov5 repository is complete.")
            # 代替として、画像をダウンロードするロジックを追加することも可能ですが、今回はエラーとして扱います。
            raise FileNotFoundError(f"Sample image not found at {sample_image_path}")

        image = Image.open(sample_image_path).convert('RGB')
        image_np_for_inference = np.array(image)

        print("--- Debugging Image and Model Parameters before Inference ---")
        print(f"Image numpy array shape: {image_np_for_inference.shape}")
        print(f"Image numpy array dtype: {image_np_for_inference.dtype}")
        print(f"Image numpy array min value: {image_np_for_inference.min()}")
        print(f"Image numpy array max value: {image_np_for_inference.max()}")
        print(f"Model confidence threshold (model.conf): {model.conf}")
        if hasattr(model, 'iou'):
            print(f"Model IoU threshold (model.iou): {model.iou}")
        else:
            print("Model has no 'iou' attribute (using default).")
        print(f"Model training mode (model.training): {model.training}")
        print("--- End Debugging Image and Model Parameters before Inference ---")

        # 物体検出を実行
        results = model(image_np_for_inference)

        print("--- Debugging Detection Results ---")
        print(f"Raw detection results: {results}")
        if hasattr(results, 'pred') and len(results.pred) > 0:
            print(f"Number of detected objects: {len(results.pred[0])}")
            for *xyxy, conf, cls in results.pred[0]:
                label = model.names[int(cls)]
                print(f"  Detected: {label} (Conf: {conf:.2f})")
        else:
            print("No 'pred' attribute or empty in detection results. No objects detected.")
        print("--- End Debugging Detection Results ---")

    except Exception as e:
        print(f"Error in debug_detection.py: {e}")
