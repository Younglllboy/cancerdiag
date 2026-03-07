import torch
import numpy as np  # 必须导入numpy
# ---------------------- 核心添加：允许numpy scalar全局对象 ----------------------
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_score  # 新增：精确率计算
)
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
# 2. 解决负号显示异常（可选，避免后续有负数时显示□）
plt.rcParams['axes.unicode_minus'] = False
# 3. 关闭字体缺失的警告（彻底消除UserWarning）
plt.rcParams['font.family'] = 'sans-serif'
import config
from model.se_resnet import SEResNet50

# ---------------------- 核心修改1：解决num_workers=4多进程报错 ----------------------
if __name__ == '__main__':
    # 数据预处理（不变）
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载测试集（num_workers=4，通过main保护块解决多进程报错）
    test_dataset = datasets.ImageFolder(
        root=os.path.join(config.DATASET_SPLIT_ROOT, 'test'),
        transform=transform
    )
    # 关键：num_workers=4 + 添加pin_memory（加速GPU数据传输，可选）
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,  # 改为4，通过main保护块避免Windows多进程报错
        pin_memory=True if config.DEVICE == 'cuda' else False  # GPU环境下加速
    )

    # 加载模型（添加weights_only=False，适配PyTorch 2.6+）
    model = SEResNet50().to(config.DEVICE)
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 推理获取结果（不变）
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[:, 1]  # 1=恶性的概率

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    # ---------------------- 核心修改2：完善恶性数据的混淆矩阵分析 ----------------------
    # 1. 基础指标（保留原有）
    acc = accuracy_score(all_labels, all_preds)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs)

    # 2. 新增：恶性（阳性）专属指标（临床更关注）
    # 类别映射：假设test_dataset.classes中 0=良性(benign)，1=恶性(malignant)
    class_names = test_dataset.classes
    malignant_label = 1 if 'malignant' in class_names else 0  # 自动匹配恶性标签
    benign_label = 0 if malignant_label == 1 else 1

    # 恶性的召回率（TPR：真阳性率，漏诊率=1-TPR）
    recall_malignant = recall_score(all_labels, all_preds, pos_label=malignant_label)
    # 恶性的精确率（PPV：阳性预测值，误诊率=1-PPV）
    precision_malignant = precision_score(all_labels, all_preds, pos_label=malignant_label)
    # 恶性的F1值（兼顾漏诊和误诊）
    f1_malignant = f1_score(all_labels, all_preds, pos_label=malignant_label)

    # 3. 混淆矩阵拆解（明确恶性相关数值）
    cm = confusion_matrix(all_labels, all_preds)
    # 混淆矩阵元素定义：
    # cm[真标签][预测标签]
    # TN: 真良性（实际良性，预测良性）= cm[benign_label][benign_label]
    # FP: 假恶性（实际良性，预测恶性）= cm[benign_label][malignant_label]
    # FN: 假良性（实际恶性，预测良性）= cm[malignant_label][benign_label] 【漏诊恶性】
    # TP: 真恶性（实际恶性，预测恶性）= cm[malignant_label][malignant_label]
    TN = cm[benign_label][benign_label]
    FP = cm[benign_label][malignant_label]
    FN = cm[malignant_label][benign_label]
    TP = cm[malignant_label][malignant_label]

    # 4. 打印完善后的指标（重点突出恶性数据）
    print("=" * 50)
    print("模型测试集评估结果（重点：恶性数据）")
    print("=" * 50)
    # 基础指标
    print(f"整体准确率 (Accuracy): {acc:.4f}")
    print(f"整体加权召回率 (Recall-weighted): {recall_weighted:.4f}")
    print(f"整体加权F1值 (F1-weighted): {f1_weighted:.4f}")
    print(f"AUC (区分良/恶性能力): {auc:.4f}")
    print("-" * 30)
    # 恶性专属指标（临床核心）
    print(f"恶性召回率 (TPR，不漏诊率): {recall_malignant:.4f} (漏诊率: {1 - recall_malignant:.4f})")
    print(f"恶性精确率 (PPV，不错诊率): {precision_malignant:.4f} (误诊率: {1 - precision_malignant:.4f})")
    print(f"恶性F1值 (兼顾漏诊/误诊): {f1_malignant:.4f}")
    print("-" * 30)
    # 混淆矩阵数值解读
    print("混淆矩阵详细拆解（恶性相关）：")
    print(f"真良性 (TN): {TN} 例（实际良性，预测良性）")
    print(f"假恶性 (FP): {FP} 例（实际良性，预测恶性）→ 良性被误诊为恶性")
    print(f"假良性 (FN): {FN} 例（实际恶性，预测良性）→ 恶性被漏诊为良性【临床高危】")
    print(f"真恶性 (TP): {TP} 例（实际恶性，预测恶性）")
    print("\n原始混淆矩阵：")
    print(cm)

    # 5. 绘制完善的混淆矩阵图（标注良/恶性，突出恶性数据）
    plt.figure(figsize=(10, 8))
    # 热力图标注：添加数值+良/恶性标签
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['良性', '恶性'],  # 中文标注，更直观
        yticklabels=['良性', '恶性'],
        annot_kws={"size": 12}  # 字体大小
    )
    # 标注恶性相关的关键区域
    plt.text(
        malignant_label + 1.1, benign_label + 1,
        f'漏诊: {FN}例',
        fontsize=10,
        color='red',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5)
    )
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('混淆矩阵（重点：恶性漏诊/误诊）', fontsize=14)
    plt.tight_layout()  # 防止标签被截断
    plt.savefig(os.path.join(config.BASE_DIR, 'model', 'confusion_matrix_malignant.png'))
    plt.show()