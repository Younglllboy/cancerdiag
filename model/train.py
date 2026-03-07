import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
os.environ["PYTORCH_MULTIPROCESSING_MODE"] = "spawn"
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import config
from model.se_resnet import SEResNet50


# ---------------------- 1. 定义训练/验证函数（放在全局）----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, recall, f1


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[:, 1]

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, recall, f1, auc


# ---------------------- 2. 主训练逻辑放入保护块（核心修改）----------------------
if __name__ == '__main__':
    # 数据预处理与增强（移到保护块内）
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集（移到保护块内）
    train_dataset = datasets.ImageFolder(
        root=os.path.join(config.DATASET_SPLIT_ROOT, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(config.DATASET_SPLIT_ROOT, 'val'),
        transform=val_test_transform
    )

    # 保留多进程（num_workers=4 可根据CPU核心数调整）
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 初始化模型、损失函数、优化器（移到保护块内）
    model = SEResNet50().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 早停机制（移到保护块内）
    best_val_auc = 0.0
    patience_counter = 0

    # 训练主循环（移到保护块内）
    print("开始训练模型...")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        train_loss, train_acc, train_recall, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer,
                                                                        config.DEVICE)
        val_loss, val_acc, val_recall, val_f1, val_auc = validate(model, val_loader, criterion, config.DEVICE)

        scheduler.step(val_loss)

        print(
            f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, 召回率: {train_recall:.4f}, F1值: {train_f1:.4f}")
        print(
            f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, 召回率: {val_recall:.4f}, F1值: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc
            }, config.MODEL_SAVE_PATH)
            print(f"保存最佳模型（AUC: {best_val_auc:.4f}）")
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{config.PATIENCE}")
            if patience_counter >= config.PATIENCE:
                print("早停机制触发，停止训练！")
                break

    print("训练完成！")