采用BreaKHis_v1开源数据集
数据集存放位置：/dataset/BreaKHis_v1
运行split_dataset，按照7：2：1划分数据集为训练集，测试集，验证集位于/dataset/split_data
运行evaluate.py生成混淆矩阵位置：/model/con...
上传病理图片位置：app/static/uploads
热力图存储位置：app/static/uploads/heatmaps
运行/model/train.py训练模型，存储位置：app/static/model/best_model.pth
修改config.py的数据库password，创建cancerbreast数据库，建表命名为detection_records，列名参照app/models/record.py
