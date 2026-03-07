from flask import Blueprint, render_template

# 根蓝图（无前缀，对应 / 路径）
index_bp = Blueprint('index', __name__)

# 核心：根路由指向index.html
@index_bp.route('/')
def index():
    # 渲染templates/index.html
    return render_template('index.html')

