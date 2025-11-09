import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import io
import faiss 
import struct 
import gzip 
import numpy as np 
import base64 
from torchvision import transforms
from model import get_model, num_classes 
import torch.nn as nn 
import traceback # トレースバック出力用

app = Flask(__name__)

# --- グローバル変数とパス定義 ---
EMBEDDINGS = None
LABELS = None
CLASS_NAMES = None
DATASET_SIZE = 0
DATA_ROOT = os.path.join('.', 'data', 'FashionMNIST', 'raw') 

FAISS_INDEX = None 
MODEL_FOR_UPLOAD = None 

PTFILE_PATH = os.path.join('ptfile', 'embeddings.pt')
BINFILE_PATH = os.path.join('binfile', 'faiss_index.bin')
MODEL_PATH_ABS = os.path.join('pthfile', 'fashion_classifier_model.pth')

RAW_IMAGE_FILE = 'train-images-idx3-ubyte'
MNIST_HEADER_SIZE = 16 

# --- PyTorchモデルと前処理関数 ---

# テンソル変換・3ch複製・正規化のみを行うパイプライン
upload_tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    # 1ch画像を3chに複製
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    # 3chのImageNetの平均と標準偏差で正規化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


# データのロード (サーバー起動時に一度だけ実行)
def load_data():
    """embeddings.pt ファイルから特徴ベクトルとFaissインデックスをロードする"""
    global EMBEDDINGS, LABELS, CLASS_NAMES, DATASET_SIZE, FAISS_INDEX, MODEL_FOR_UPLOAD
    
    CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("--- データロード開始 ---")
    try:
        # 1. 特徴ベクトルとラベルをロード
        if not os.path.exists(PTFILE_PATH):
            raise FileNotFoundError(f"特徴ベクトルファイル '{PTFILE_PATH}' が見つかりません。")
            
        data = torch.load(PTFILE_PATH)
        EMBEDDINGS = data['embeddings'].numpy().astype('float32') 
        LABELS = data['labels'].numpy()
        DATASET_SIZE = EMBEDDINGS.shape[0]
        print(f"特徴ベクトルロード成功。サイズ: {DATASET_SIZE}")
        
        # 2. Faissインデックスをロード
        if not os.path.exists(BINFILE_PATH):
            raise FileNotFoundError(f"Faissインデックスファイル '{BINFILE_PATH}' が見つかりません。")
            
        FAISS_INDEX = faiss.read_index(BINFILE_PATH)
        print(f"Faissインデックスロード成功。次元: {FAISS_INDEX.d}")

        # 3. 特徴抽出モデルのロード (アップロード処理用)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(MODEL_PATH_ABS):
            raise FileNotFoundError(f"モデル重みファイル '{MODEL_PATH_ABS}' が見つかりません。")
            
        # feature_extractor_only=True で4096次元を出力するモデルをロード
        MODEL_FOR_UPLOAD = get_model(num_classes=num_classes, feature_extractor_only=True)
        
        # 学習済み重みのロード
        state_dict = torch.load(MODEL_PATH_ABS, map_location=device, weights_only=True)

        # 最終層の重みが不足するため、strict=Falseで無視する
        MODEL_FOR_UPLOAD.load_state_dict(state_dict, strict=False)
        MODEL_FOR_UPLOAD.to(device)
        MODEL_FOR_UPLOAD.eval() 
        
        print(f"特徴抽出モデルロード成功: '{MODEL_PATH_ABS}' on {device}")
        
        return True
    except FileNotFoundError as e:
        print(f"エラー: 必要なファイルが見つかりません。学習と特徴抽出を実行してください: {e}")
        return False
    except Exception as e:
        print(f"データロード中にエラーが発生しました: {e}") 
        return False

# 画像データ取得ヘルパー
def get_image_data(index):
    file_path = os.path.join(DATA_ROOT, 'train-images-idx3-ubyte')
    
    if not os.path.exists(file_path):
        compressed_path = file_path + '.gz'
        if os.path.exists(compressed_path):
             with gzip.open(compressed_path, 'rb') as f_in:
                with open(file_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            raise FileNotFoundError(f"画像ファイルが見つかりません: {file_path} または {compressed_path}")
            
    image_size = 28 * 28
    MNIST_HEADER_SIZE = 16 
    offset = MNIST_HEADER_SIZE + (index * image_size)
    
    with open(file_path, 'rb') as f:
        f.seek(offset)
        image_bytes = f.read(image_size)
    
    image = Image.frombytes('L', (28, 28), image_bytes) 
    return image
    
# 推薦ロジック
def get_recommendations(target_index, num_recommendations=5):
    
    target_embedding = EMBEDDINGS[target_index:target_index+1] 
    K = num_recommendations + 1
    
    Distances, Indices = FAISS_INDEX.search(target_embedding, K)
    
    recommendations = []
    
    for i in range(K):
        index = Indices[0, i]
        
        if index == target_index:
            continue
            
        distance = Distances[0, i]
        similarity = 1 - distance / 2.0 

        recommendations.append({
            'index': int(index),
            'label': int(LABELS[index]),
            'class_name': CLASS_NAMES[LABELS[index]],
            'similarity': float(similarity), 
            'image_url': f'/image/{index}' 
        })
        
        if len(recommendations) >= num_recommendations:
            break
            
    return recommendations


# Flask ルート
@app.before_request
def before_first_request():
    if FAISS_INDEX is None:
        load_data()


@app.route('/')
def index():
    import random
    if DATASET_SIZE > 0:
        initial_index = random.randint(0, DATASET_SIZE - 1)
    else:
        initial_index = 0
        
    return render_template('index.html', initial_index=initial_index)

@app.route('/image/<int:index>')
def get_image(index):
    if 0 <= index < DATASET_SIZE:
        try:
            image = get_image_data(index)
            
            img_io = io.BytesIO()
            image.resize((224, 224), resample=Image.Resampling.NEAREST).save(img_io, 'PNG') 
            img_io.seek(0)
            
            return img_io.getvalue(), 200, {'Content-Type': 'image/png'}
        except FileNotFoundError as e:
             return f"画像ファイルが見つかりません: {e}", 404
        except Exception as e:
            return f"画像の取得中にエラーが発生しました: {e}", 500
    else:
        return "無効な画像インデックスです", 404


@app.route('/recommend/<int:target_index>', methods=['GET'])
def recommend_api(target_index):
    if FAISS_INDEX is None:
        return jsonify({'error': 'データまたはFaissインデックスがロードされていません'}), 500
        
    if 0 <= target_index < DATASET_SIZE:
        
        target_item = {
            'index': target_index,
            'label': int(LABELS[target_index]),
            'class_name': CLASS_NAMES[LABELS[target_index]],
            'image_url': f'/image/{target_index}'
        }
        
        recommendations = get_recommendations(target_index, num_recommendations=5) 

        return jsonify({
            'target': target_item,
            'recommendations': recommendations
        })
    else:
        return jsonify({'error': '無効なインデックスが指定されました'}), 400


# アップロード画像の特徴抽出と推薦ルート
@app.route('/upload_recommend', methods=['POST'])
def upload_recommend_api():
    if MODEL_FOR_UPLOAD is None or FAISS_INDEX is None:
        return jsonify({'error': 'サーバー側の初期化が完了していません (モデル/Faiss)'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'ファイルがアップロードされていません'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'ファイル名が空です'}), 400

    try:
        # 画像の読み込みと前処理 (堅牢なPIL処理)
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # RGBモードに強制変換（様々な画像形式に対応）
        img_copy = img.convert('RGB')
        
        # VGG16の入力サイズにリサイズ
        img_resized = img_copy.resize((224, 224), resample=Image.Resampling.NEAREST)
        
        # グレースケールに変換（Fashion-MNISTで転移学習しているため）
        img_gray = img_resized.convert('L') 
        
        # ンソル変換・3ch複製・正規化を実行
        input_tensor_3ch = upload_tensor_transform(img_gray)
        
        # モデルの入力に合わせてバッチ次元を追加 [1, 3, 224, 224]
        inputs_to_model = input_tensor_3ch.unsqueeze(0) 

        # デバイスへの移動と特徴抽出
        device = next(MODEL_FOR_UPLOAD.parameters()).device 
        inputs_to_model = inputs_to_model.to(device)

        with torch.no_grad():
            # 4096次元の特徴ベクトルを取得
            features = MODEL_FOR_UPLOAD(inputs_to_model) 
            upload_embedding = features.cpu().numpy().astype('float32')
            
        # Faissでの検索
        faiss.normalize_L2(upload_embedding) 
        K = 5 
        
        # 次元が4096次元で一致しているため、検索が成功する
        Distances, Indices = FAISS_INDEX.search(upload_embedding, K)
        
        # 結果の整形
        recommendations = []
        for i in range(0, K): 
            index = Indices[0, i]
            distance = Distances[0, i]
            similarity = 1 - distance / 2.0 

            recommendations.append({
                'index': int(index),
                'label': int(LABELS[index]),
                'class_name': CLASS_NAMES[LABELS[index]],
                'similarity': float(similarity), 
                'image_url': f'/image/{index}' 
            })

        # ターゲット画像のデータURLを生成
        img_io = io.BytesIO()
        # リサイズ済みのカラー画像をデータURLに含める
        img_resized.save(img_io, 'PNG') 
        img_data_url = "data:image/png;base64," + base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        target_item = {
            'index': 'UPLOAD', 
            'class_name': 'アップロード画像',
            'image_url': img_data_url 
        }

        return jsonify({
            'target': target_item,
            'recommendations': recommendations
        })

    except Exception as e:
        print("\n--- アップロード画像処理エラー詳細 ---")
        traceback.print_exc() 
        print("----------------------------------\n")
        
        error_message = f"{type(e).__name__}: {e}"
        return jsonify({'error': f'画像処理中にエラーが発生しました: {error_message}'}), 500


if __name__ == '__main__':
    if load_data():
        app.run(debug=True, threaded=False)
    else:
        print("データロードに失敗したため、サーバーを起動できません。")