import sys
import os
import torch
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

    # yolov5s.ptのダウンロード
    if not os.path.exists(model_weights_path):
        print(f"Downloading yolov5s.pt to {model_weights_path}")
        os.makedirs(os.path.dirname(model_weights_path), exist_ok=True)
        url = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt"
        try:
            r = requests.get(url, allow_redirects=True, timeout=10) # タイムアウトを10秒に設定
            r.raise_for_status() # HTTPエラーが発生した場合に例外を発生させる
            with open(model_weights_path, 'wb') as f:
                f.write(r.content)
            print("yolov5s.pt downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading yolov5s.pt: {e}")
            print(f"Please check your internet connection or the URL: {url}")
            raise # エラーを再発生させて処理を中断
        except IOError as e:
            print(f"Error writing yolov5s.pt to disk: {e}")
            print(f"Please check write permissions for {model_weights_path}")
            raise # エラーを再発生させて処理を中断
        except Exception as e:
            print(f"An unexpected error occurred during yolov5s.pt download: {e}")
            raise # エラーを再発生させて処理を中断

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
    print("Attempting to load model from debug_model_loading.py...")
    try:
        model = load_model()
        print("Model loaded successfully in debug script.")
    except Exception as e:
        print(f"Error loading model in debug script: {e}")
