import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time
import copy 
import numpy as np 

# --- パス定義 ---
MODEL_DIR = 'pthfile'
MODEL_FILENAME = 'fashion_classifier_model.pth'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

#  データセットとデータローダーの定義 (変更なし)
data_transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

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


# PyTorchモデルの定義

def get_model(num_classes, feature_extractor_only=False):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # 畳み込み層の重みを凍結 
    for param in model.features.parameters():
        param.requires_grad = False
    
    # 分類器層を再構築
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    
    if feature_extractor_only:
        model.classifier = nn.Sequential(*model.classifier[:4]) 
    
    return model

# 3. 特徴抽出関数
def extract_embeddings(model, dataloader, device):
    model.eval() 
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for inputs, batch_labels in dataloader:
            inputs = inputs.to(device)
        
            # get_modelで分類器を切り詰めたため、ここで4096次元が出力される
            features = model(inputs) 
            
            embeddings.append(features.cpu())
            labels.append(batch_labels)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeddings, labels


# モデル学習関数
def train_model(model, dataloader, criterion, optimizer, num_epochs=3):
    since = time.time()
    device = torch.device("cuda:0") 
    model.to(device)
    
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

# メイン処理
if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True) 
    device = torch.device("cuda:0") 

    model_for_training = get_model(num_classes, feature_extractor_only=False)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model_for_training.classifier.parameters(), lr=0.01, momentum=0.9)

    print("--- 学習開始 ---")
    model_for_training = train_model(model_for_training, all_loader, criterion, optimizer, num_epochs=3) 
    print("--- 学習完了 ---")

    torch.save(model_for_training.state_dict(), MODEL_PATH)
    print(f"モデルを '{MODEL_PATH}' として保存しました。")