import torch

# YOLOv5公式モデルをPyTorch 2.7.0で読み込み
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 新しい形式で保存
torch.save(model, 'yolov5s.pt')
print("新しいモデルファイルを保存しました： yolov5s.pt")