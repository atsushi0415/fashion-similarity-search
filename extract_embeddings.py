import torch
import torch.nn as nn
from model import get_model, all_loader, extract_embeddings, num_classes 
import os
import sys
import faiss
import numpy as np 

# --- パス定義 ---
MODEL_PATH = os.path.join('pthfile', 'fashion_classifier_model.pth')
PTFILE_PATH = os.path.join('ptfile', 'embeddings.pt')
BINFILE_PATH = os.path.join('binfile', 'faiss_index.bin')

# モデルの初期化と重みのロード
device = torch.device("cuda:0") 
print(f"特徴抽出デバイス: {device}")

model_for_extraction = get_model(num_classes=num_classes, feature_extractor_only=True)
model_for_extraction.to(device) 

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"モデルファイル '{MODEL_PATH}' が見つかりません。")
        
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model_for_extraction.load_state_dict(state_dict, strict=False)
    print(f"'{MODEL_PATH}' から学習済み重みをロードしました。")
    
    # 特徴抽出の実行
    print("--- 4096次元特徴ベクトル抽出開始 ---")
    embeddings, labels = extract_embeddings(model_for_extraction, all_loader, device)

    # 結果の保存とFaiss IndexIVFFlatの構築
    os.makedirs('ptfile', exist_ok=True)
    os.makedirs('binfile', exist_ok=True)

    embeddings_np = embeddings.numpy().astype('float32') 
    
    # L2正規化
    faiss.normalize_L2(embeddings_np)
    
    d = embeddings_np.shape[1] # 4096次元
    
    NLIST = 100
    
    quantizer = faiss.IndexFlatL2(d) 
    
    index = faiss.IndexIVFFlat(quantizer, d, NLIST, faiss.METRIC_L2)
    
    # トレーニングが必要
    print(f"IndexIVFFlat (NLIST={NLIST}) のトレーニングを開始します...")
    index.train(embeddings_np) 
    print("トレーニング完了。")
    
    # 特徴ベクトルをインデックスに追加
    index.add(embeddings_np)
    print(f"全アイテム ({embeddings_np.shape[0]}点) を IndexIVFFlat に追加しました。")
    
    # Faissをファイルに保存
    faiss.write_index(index, BINFILE_PATH)
    print(f"Faissインデックス（IndexIVFFlat, {d}次元）を '{BINFILE_PATH}' として保存しました。")
    
    # 正規化済みの特徴ベクトルをembeddings.ptに保存
    torch.save({
        'embeddings': torch.from_numpy(embeddings_np), 
        'labels': labels,
        'class_names': all_loader.dataset.classes, 
        'dataset_size': len(all_loader.dataset)
    }, PTFILE_PATH)

    print(f"全アイテムの特徴ベクトルを '{PTFILE_PATH}' として保存しました。")
    
except FileNotFoundError as e:
    print(f"エラー: {e}\nmodel.pyの学習を完了させ、必要なファイルを生成してください。", file=sys.stderr)
except Exception as e:
    print(f"特徴抽出中にエラーが発生しました: {e}", file=sys.stderr)