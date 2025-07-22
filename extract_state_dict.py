import torch
import os

# PyTorch 2.6+ のセキュリティ変更に対応
try:
    from yolov5.models.common import AutoShape
    torch.serialization.add_safe_globals([AutoShape])
except (ImportError, AttributeError):
    pass

# モデルファイルのパス
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5s.pt')

# モデルを読み込み、state_dictを抽出
# weights_only=False を指定して、古い形式のモデルも読み込めるようにする
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

# AutoShapeラッパーから実際のモデルのstate_dictを取得
torch.save(model.model.state_dict(), 'yolov5s_state_dict.pt')
print("モデルのstate_dictを保存しました： yolov5s_state_dict.pt")