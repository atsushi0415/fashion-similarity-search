import torch
import torch.nn as nn
from model import get_model, all_loader, extract_embeddings, num_classes 
import os
import sys
import faiss

# モデルの初期化と重みのロード
MODEL_PATH = 'fashion_classifier_model.pth'

# GPU/CPUの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"特徴抽出デバイス: {device}")

# モデル構造を定義
model_for_extraction = get_model(num_classes=num_classes, feature_extractor_only=False)
model_for_extraction.to(device)

try:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model_for_extraction.load_state_dict(state_dict)
    print(f"'{MODEL_PATH}' から学習済み重みをロードしました。")
    
    # 特徴抽出の実行
    print("--- GPU/CPUで特徴ベクトル抽出開始 ---")
    embeddings, labels = extract_embeddings(model_for_extraction, all_loader, device)

    # 結果の保存
    # Faissで効率的に検索できるように、特徴ベクトルをL2正規化
    embeddings_np = embeddings.numpy().astype('float32') 
    
    # L2正規化
    faiss.normalize_L2(embeddings_np)
    
    # Faissインデックスを構築 
    d = embeddings_np.shape[1] 
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    
    # Faissをファイルに保存
    faiss.write_index(index, 'faiss_index.bin')
    print("Faissインデックスを 'faiss_index.bin' として保存しました。")
    
    # NumPyに変換された正規化済みの特徴ベクトルをembeddings.ptに保存
    torch.save({
        'embeddings': torch.from_numpy(embeddings_np), 
        'labels': labels,
        'class_names': all_loader.dataset.classes, 
        'dataset_size': len(all_loader.dataset)
    }, 'embeddings.pt')

    print(f"全アイテムの特徴ベクトル ({embeddings_np.shape[0]}点, {embeddings_np.shape[1]}次元) を 'embeddings.pt' として保存しました。")
    
except FileNotFoundError:
    print(f"エラー: 重みファイル '{MODEL_PATH}' が見つかりません。model.pyの学習を完了させてください。", file=sys.stderr)
except Exception as e:
    print(f"特徴抽出中にエラーが発生しました: {e}", file=sys.stderr)