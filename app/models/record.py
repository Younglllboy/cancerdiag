
from app import db
from datetime import datetime

#ORM
class DetectionRecord(db.Model):
    """
    乳腺癌病理图像检测记录模型
    存储上传图片的文件名、诊断结果、置信度、上传时间等信息
    """
    # 显式指定数据库表名（符合命名规范，无需修改，全新项目直接用）
    __tablename__ = 'detection_records'

    # 实体关系映射（DetectionRecord，detection_records与代码中实例化参数完全匹配，避免invalid keyword报错）
    #id主键自增无需人工赋值
    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment='记录ID')
    filename = db.Column(db.String(255), nullable=False, comment='上传的图片文件名')
    result = db.Column(db.String(10), nullable=False, comment='诊断结果：良性/恶性')
    confidence = db.Column(db.Float, nullable=False, comment='诊断置信度（0-1）')
    upload_time = db.Column(db.DateTime, default=datetime.now, comment='上传/诊断时间')

    def __repr__(self):
        """自定义打印格式，便于调试"""
        return f"<DetectionRecord {self.id}: {self.filename} - {self.result} (置信度: {self.confidence:.4f})>"

    def to_dict(self):
        """将模型转为字典，便于前端序列化（可选扩展）"""
        # 生成病理图片URL（与上传时的规则一致）
        image_url = f'../static/uploads/{self.filename}'
        # 生成热力图URL（热力图命名规则：heatmap_ + 原文件名）
        heatmap_filename = f'heatmap_{self.filename}'
        heatmap_url = f'../static/uploads/heatmaps/{heatmap_filename}'
        return {
            'id': self.id,
            'filename': self.filename,
            'result': self.result,
            'confidence': round(self.confidence, 4),
            'upload_time': self.upload_time.strftime('%Y-%m-%d %H:%M:%S'),
            # 核心新增：图片URL字段
            'image_url': image_url,
            'heatmap_url': heatmap_url
        }