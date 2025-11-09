import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy 
import numpy as np 

#データセットとデータローダーの定義

# data_transformの定義
data_transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Fashion-MNISTデータセットのダウンロード (app.pyとextract_embeddings.pyでデータロードに利用)
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=data_transform)
all_dataset = train_dataset

all_loader = torch.utils.data.DataLoader(
    all_dataset, 
    batch_size=64, 
    shuffle=False,
    num_workers=0
)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_classes = len(class_names)


# yTorchモデルの定義（特徴抽出器として再定義）
def get_model(num_classes=10, feature_extractor_only=False):
    # ImageNetで事前学習済みのVGG16をロード
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # 畳み込み層の重みを凍結 
    for param in model.features.parameters():
        param.requires_grad = False
    
    # 分類器層を再構築（最終層まで含んだフルモデル）
    model.classifier = nn.Sequential(
        # 100352 (VGG16 feature map size) -> 4096
        nn.Linear(model.classifier[0].in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    
    # 特徴抽出のみを行う場合、最終層（分類層）を削除
    if feature_extractor_only:
        model.classifier = model.classifier[:4] 
    return model

#特徴ベクトル抽出関数
def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # inputsはdata_transformにより既に3ch化されているため、そのまま渡す
            features = model.features(inputs)
            flattened_features = features.view(features.size(0), -1) 
            
            features_4096 = model.classifier[:4](flattened_features) 
            
            embeddings.append(features_4096.cpu())
            labels.append(targets.cpu())
    
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

# モデル学習関数
def train_model(model, dataloader, criterion, optimizer, num_epochs=3):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    

    
    # 簡易のため、train_modelのデータローダーはall_loaderを使用
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f}')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model

# メイン処理 (学習済みモデルの確認と保存)
if __name__ == '__main__':
    MODEL_PATH = 'fashion_classifier_model.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # モデルの学習と保存
    model_for_training = get_model(num_classes, feature_extractor_only=False)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model_for_training.classifier.parameters(), lr=0.01, momentum=0.9)

    print("--- 学習開始 ---")
    model_for_training = train_model(model_for_training, all_loader, criterion, optimizer, num_epochs=3) 
    print("--- 学習完了 ---")

    torch.save(model_for_training.state_dict(), MODEL_PATH)
    print(f"モデルを '{MODEL_PATH}' として保存しました。")