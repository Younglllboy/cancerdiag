from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import config
import os

# 初始化数据库
db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    # 配置项
    app.config['SECRET_KEY'] = 'breast_cancer_detection_secret_key'
    app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.SQLALCHEMY_TRACK_MODIFICATIONS
    app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小16MB

    # 创建上传目录和热力图目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(config.HEATMAP_FOLDER, exist_ok=True)

    # 初始化数据库
    db.init_app(app)


    # ---------------------- 延迟导入蓝图（核心解决循环导入） ----------------------
    from app.routes.index import index_bp
    from app.routes.upload import upload_bp
    from app.routes.records import records_bp
    # ---------------------- 注册蓝图 ----------------------
    app.register_blueprint(index_bp)  # 根路由 /
    app.register_blueprint(upload_bp)  # /upload/*
    app.register_blueprint(records_bp)  # /records/*

    # 创建数据库表（首次运行时）
    with app.app_context():
        db.create_all()

    return app