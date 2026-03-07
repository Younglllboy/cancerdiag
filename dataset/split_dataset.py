import os
import shutil
from sklearn.model_selection import train_test_split
import config


def create_dir_if_not_exist(path):
    """创建目录（不存在则创建）"""
    if not os.path.exists(path):
        os.makedirs(path)


# 初始化划分后的数据目录
split_root = config.DATASET_SPLIT_ROOT
create_dir_if_not_exist(split_root)
for subset in ['train', 'val', 'test']:
    for label in ['benign', 'malignant']:
        create_dir_if_not_exist(os.path.join(split_root, subset, label))

# 遍历所有类别，划分数据
for label in ['benign', 'malignant']:
    label_dir = os.path.join(config.DATASET_RAW_ROOT, label)
    image_paths = []

    # 收集该类别下所有PNG图片
    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.png'):
                image_paths.append(os.path.join(root, file))

    # 划分训练集（70%）、验证集（10%）、测试集（20%）
    train_paths, temp_paths = train_test_split(image_paths, test_size=0.3, random_state=42, shuffle=True)
    val_paths, test_paths = train_test_split(temp_paths, test_size=0.6667, random_state=42, shuffle=True)


    # 复制图片到对应目录
    def copy_images(paths, subset):
        for src_path in paths:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(split_root, subset, label, filename)
            shutil.copyfile(src_path, dst_path)


    copy_images(train_paths, 'train')
    copy_images(val_paths, 'val')
    copy_images(test_paths, 'test')

print("数据集划分完成！")
print(f"训练集数量：{len(train_paths)}")
print(f"验证集数量：{len(val_paths)}")
print(f"测试集数量：{len(test_paths)}")