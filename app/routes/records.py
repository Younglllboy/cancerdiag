from flask import Blueprint, jsonify
from app.models.record import DetectionRecord
import os
import config
from app import db
records_bp = Blueprint('records', __name__, url_prefix='/records')

@records_bp.route('/', methods=['GET'])
def get_records():
    """查询所有检测记录（按时间倒序）"""
    records = DetectionRecord.query.order_by(DetectionRecord.upload_time.desc()).all()
    return jsonify({
        'code': 200,
        'data': [record.to_dict() for record in records]
    })

@records_bp.route('/<int:record_id>', methods=['GET'])
def get_record_detail(record_id):
    """查询单个检测记录详情"""
    record = DetectionRecord.query.get(record_id)
    if not record:
        return jsonify({'code': 404, 'msg': '记录不存在'}), 404
    return jsonify({
        'code': 200,
        'data': record.to_dict()
    })

@records_bp.route('/<int:record_id>', methods=['DELETE'])
def del_record_detail(record_id):
    """
    删除单条检测记录
    - 先删除数据库记录
    - 再删除对应的病理图片和热力图文件（避免文件残留）
    - 包含完整的异常处理和事务回滚
    """
    try:
     record = DetectionRecord.query.get(record_id)
     if not record:
        return jsonify({'code': 404, 'msg': '记录不存在'}), 404
     #删除数据库中的记录（核心操作）
     db.session.delete(record)
     #提交事务（确认删除）
     db.session.commit()
     print(f"已删除数据库记录：ID={record_id}，文件名={record.filename}")

     # 2.1 拼接病理图片和热力图的完整路径
     # 病理图片路径（和上传时的保存路径一致）
     image_path = os.path.join(config.UPLOAD_FOLDER, record.filename)
     # 热力图路径（命名规则：heatmap_ + 原文件名）
     heatmap_filename = f'heatmap_{record.filename}'
     heatmap_path = os.path.join(config.HEATMAP_FOLDER, heatmap_filename)

     # 2.2 执行文件删除（容错：文件不存在时不报错）
     # 删除病理图片
     if os.path.exists(image_path):
        os.remove(image_path)
        print(f" 已删除病理图片：{image_path}")
     # 删除热力图
     if os.path.exists(heatmap_path):
        os.remove(heatmap_path)
        print(f" 已删除热力图：{heatmap_path}")

     #返回成功响应
     return jsonify({
         'code': 200,
         'msg': f'ID为{record_id}的记录及对应图片已成功删除'
     }), 200
    except Exception as e:
        # 异常处理：事务回滚 + 返回错误信息
        # 关键：如果出错，回滚数据库操作，避免数据不一致
        db.session.rollback()
        # 打印错误日志（便于调试）
        print(f" 删除记录失败：{str(e)}")
        # 返回500错误，给前端友好提示
        return jsonify({
            'code': 500,
            'msg': f'删除失败：{str(e)}'
        }), 500