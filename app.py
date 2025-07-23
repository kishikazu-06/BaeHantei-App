import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from io import BytesIO
import sys
import os
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

@st.cache_resource
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
        model.eval() # モデルを評価モードに設定

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

def calculate_brightness_score(image_np):
    #画像の明るさを評価
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)  # 0-255
    return brightness / 2.55  # 0-100のスコアに正規化

def calculate_saturation_score(image_np):
    #画像の色彩の鮮やかさ（彩度）を評価し、0-100のスコアを返す
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    saturation = np.mean(hsv[..., 1])  # Sチャネル (0-255)
    return saturation / 2.55  # 0-100に正規化

def calculate_center_composition_score(detections, image_shape):
    #被写体が中央に配置されている構図（日の丸構図）を評価する
    h, w, _ = image_shape
    center_x, center_y = w / 2, h / 2
    
    if len(detections.pred[0]) == 0:
        return 0

    total_proximity = 0
    for *xyxy, conf, cls in detections.pred[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        obj_cx = (x1 + x2) / 2
        obj_cy = (y1 + y2) / 2
        distance = np.sqrt((obj_cx - center_x)**2 + (obj_cy - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        proximity = 1 - (distance / max_distance)
        total_proximity += proximity

    avg_proximity = total_proximity / len(detections.pred[0])
    return avg_proximity * 100

def calculate_rule_of_thirds_score(detections, image_shape):
    #三分割法に基づいた構図を評価する
    h, w, _ = image_shape
    thirds_x = [w / 3, 2 * w / 3]
    thirds_y = [h / 3, 2 * h / 3]
    
    if len(detections.pred[0]) == 0:
        return 0

    max_score = 0
    for *xyxy, conf, cls in detections.pred[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        obj_cx = (x1 + x2) / 2
        obj_cy = (y1 + y2) / 2

        min_dist_x = min(abs(obj_cx - tx) for tx in thirds_x)
        min_dist_y = min(abs(obj_cy - ty) for ty in thirds_y)

        score_x = max(0, 100 - (min_dist_x / (w / 6)) * 100)
        score_y = max(0, 100 - (min_dist_y / (h / 6)) * 100)
        
        current_score = (score_x + score_y) / 2
        
        if current_score > max_score:
            max_score = current_score
            
    return min(max_score, 100)

def calculate_composition_score(detections, image_shape):
    #中心構図と三分割法を評価し、良い方のスコアを返す
    center_score = calculate_center_composition_score(detections, image_shape)
    thirds_score = calculate_rule_of_thirds_score(detections, image_shape)
    return max(center_score, thirds_score)

def calculate_instagenic_score(detections):
    #「映え」やすい被写体を評価する
    instagenic_objects = {
        "person": 10, "cat": 20, "dog": 20, "bird": 15,
        "cake": 25, "pizza": 20, "donut": 20, "wine glass": 15,
        "car": 15, "bicycle": 10, "boat": 15,
        "bench": 5, "handbag": 10, "suitcase": 5,
        "sports ball": 10, "surfboard": 20,
        # YOLOv5のデフォルトクラスに合わせて調整
    }
    
    detected_classes = [detections.names[int(cls)] for cls in detections.pred[0][:, -1].cpu().numpy()]
    score = 0

    print("--- Debugging Instagenic Score ---")
    print(f"Detected classes: {detected_classes}")

    for obj in set(detected_classes): # 重複を除外して加算
        if obj in instagenic_objects:
            score += instagenic_objects[obj]
            print(f"  - {obj}: +{instagenic_objects[obj]} points")
            
    print(f"Total instagenic score before min(100): {score}")
    print("--- End Debugging Instagenic Score ---")

    return min(score, 100) # スコアの上限を100に設定

def calculate_total_sns_score(image, detections):
    #各評価項目のスコアを計算し、重み付けして総合スコアを算出する
    image_np = np.array(image)
    
    # --- 各項目を0-100点で評価 ---
    brightness_score = calculate_brightness_score(image_np)
    saturation_score = calculate_saturation_score(image_np)
    composition_score = calculate_composition_score(detections, image_np.shape)
    instagenic_score = calculate_instagenic_score(detections)

    # --- 各項目の重み付け ---
    weights = {
        "composition": 0.4,
        "saturation": 0.3,
        "instagenic": 0.2,
        "brightness": 0.1
    }

    # --- 重み付けした総合スコアを計算 ---
    total_score = (
        composition_score * weights["composition"] +
        saturation_score * weights["saturation"] +
        instagenic_score * weights["instagenic"] +
        brightness_score * weights["brightness"]
    )

    # --- 採点内訳をUIに表示 ---
    st.markdown("### 採点内訳")
    st.markdown(f"- **構図** (日の丸/三分割法): **{composition_score:.1f}** / 100 点")
    st.markdown(f"- **色彩の鮮やかさ**: **{saturation_score:.1f}** / 100 点")
    st.markdown(f"- **被写体の魅力**: **{instagenic_score:.1f}** / 100 点")
    st.markdown(f"- **全体の明るさ**: **{brightness_score:.1f}** / 100 点")

    return round(total_score)

def get_rank(score):
    #総合スコアに応じてランクを返す
    if score >= 90:
        return "S（神レベルの映え写真！）"
    elif score >= 75:
        return "A（かなり映えてます！）"
    elif score >= 60:
        return "B（良い感じの映え写真）"
    elif score >= 45:
        return "C（まあまあ映えてます）"
    else:
        return "D（もう少し工夫してみよう）"

def main():
    st.title("✨ SNS映え写真 採点アプリ ✨")
    st.write("あなたの写真がどれくらい映えるか、AIが多角的に採点します。")

    uploaded_file = st.file_uploader("ここに画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        st.image(image, caption="アップロードされた画像", use_column_width=True)

        with st.spinner('AIが画像を分析中です...'):
            model = load_model()

            # 推論前の画像データとモデルパラメータのデバッグ
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

            results = model(image_np_for_inference)

            print("--- Debugging Detection Results ---")
            print(f"Raw detection results: {results}")
            if hasattr(results, 'pred') and len(results.pred) > 0:
                print(f"Number of detected objects: {len(results.pred[0])}")
            else:
                print("No 'pred' attribute or empty in detection results.")
            print("--- End Debugging Detection Results ---")

            # 総合スコアを計算
            print("--- Calling calculate_total_sns_score ---")
            score = calculate_total_sns_score(image, results)
            print("--- Finished calculate_total_sns_score ---")
            
            st.markdown(f"## 総合SNS映えスコア: **{score}** / 100点")

            # ランク表示
            rank = get_rank(score)
            st.markdown(f"### ランク判定： **{rank}**")

            # 検出結果の画像を表示
            st.markdown("---")
            st.subheader("物体検出の結果")
            results_img = np.squeeze(results.render())
            st.image(results_img, caption="検出されたオブジェクト", use_column_width=True)

if __name__ == "__main__":
    main()
