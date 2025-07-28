import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from io import BytesIO

@st.cache_resource
def load_model():
    # YOLOv5モデルをロードしてキャッシュする
    # torch.hub.loadは、指定されたリポジトリからモデルをロードします。
    # 'ultralytics/yolov5' はGitHubリポジトリ、'yolov5s' はモデル名、pretrained=Trueは事前学習済みモデルを使用することを示します。
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    
    model.eval() # モデルを評価モードに設定
    model.conf = 0.25 # 信頼度閾値を調整
    model.iou = 0.01 # NMSのIoU閾値を調整 

    # モデルがロードされたことを確認するためのデバッグ出力
    print("Model loaded successfully using torch.hub.load.")
    print(f"Type of loaded model: {type(model)}")
    if hasattr(model, 'names'):
        print(f"Model class names: {model.names}")
    else:
        print("Model has no 'names' attribute.")
    # モデルがロードされたことを確認するためのデバッグ出力
    
    return model

def calculate_brightness_score(image_np):
    #画像の明るさを評価
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)#グレースケールに変換
    brightness = np.mean(gray)  # 0-255　グレースケール画像全体の平均輝度を計算し、画像の全体的な明るさの指標とします。
    return brightness / 2.55  # 0-100のスコアに正規化

def calculate_saturation_score(image_np):
    #画像の色彩の鮮やかさ（彩度）を評価
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)#HSVに変換
    saturation = np.mean(hsv[..., 1])  # Sチャネル (0-255)　HSV画像から彩度（S）チャネルのみを抽出し、その平均値を計算して画像の全体的な彩度を評価します。
    return saturation / 2.55  # 0-100に正規化

def calculate_center_composition_score(detections, image_shape):
    #被写体が中央に配置されている構図（日の丸構図）を評価する
    h, w, _ = image_shape #高さと幅のみを取得
    center_x, center_y = w / 2, h / 2#画像の中心座標
    
    if len(detections.pred[0]) == 0:#物体検出の有無
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
        # --- スコアを全体的に甘く調整 ---
        "person": 15, "cat": 30, "dog": 30, "bird": 20, 
        "cake": 35, "pizza": 30, "donut": 30, "wine glass": 25,
        "car": 20, "bicycle": 15, "boat": 20,
        "bench": 10, "handbag": 15, "suitcase": 10,
        "sports ball": 15, "surfboard": 30,
        
        # --- 新しく「映え」オブジェクトを追加 ---
        "apple": 15, "orange": 15, "banana": 10, "sandwich": 20, "hot dog": 20,
        "cup": 15, "fork": 10, "knife": 10, "spoon": 10, "bowl": 15,
        "bed": 10, "dining table": 20, "laptop": 10, "mouse": 5, "remote": 5,
        "keyboard": 5, "cell phone": 15, "microwave": 5, "oven": 5, "toaster": 5,
        "sink": 5, "refrigerator": 5, "book": 15, "clock": 15, "vase": 25,
        "scissors": 5, "teddy bear": 35, "potted plant": 20, "tv": 10,
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
    #総合スコアに応じてランクを返す(甘めの採点)
    if score >= 85:
        return "S（神レベルの映え写真！）"
    elif score >= 70:
        return "A（かなり映えてます！）"
    elif score >= 55:
        return "B（良い感じの映え写真）"
    elif score >= 40:
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
