import os
import torch
# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_RAW_ROOT = os.path.join(BASE_DIR, 'dataset', 'BreaKHis_v1')  # 原始数据集路径
DATASET_SPLIT_ROOT = os.path.join(BASE_DIR, 'dataset', 'split_data')  # 划分后数据集路径
MODEL_SAVE_PATH = os.path.join(BASE_DIR, '', 'app/static', 'model', 'best_model.pth')  # 模型保存路径
UPLOAD_FOLDER = os.path.join(BASE_DIR, '', 'app/static', 'uploads')  # 上传图片路径
HEATMAP_FOLDER = os.path.join(BASE_DIR, '', 'app/static', 'uploads', 'heatmaps')  # 热力图路径

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '0127',
    'database': 'cancerbreast',
    'port': 3306
}
SQLALCHEMY_DATABASE_URI = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型超参数
IMAGE_SIZE = 224  # 图像Resize尺寸
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4  # L2正则化系数
DROPOUT_RATE = 0.5
SE_REDUCTION = 16  # SE模块压缩系数
PATIENCE = 10  # 早停耐心值
NUM_CLASSES = 2  # 二分类（良性/恶性）

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备：{DEVICE}")