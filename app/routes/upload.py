from flask import Blueprint, request, jsonify, current_app
import os
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import config

# 解决PyTorch 2.6+ numpy scalar报错
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# ---------------------- grad-cam 1.4.6 专属导入（核心适配） ----------------------
from pytorch_grad_cam import GradCAM  # 1.4.6 原生支持的GradCAM类
from pytorch_grad_cam.utils.image import show_cam_on_image  # 1.4.6 热力图叠加方法
from model.se_resnet import SEResNet50

# 初始化上传蓝图
upload_bp = Blueprint('upload', __name__, url_prefix='/upload')


# 加载模型（全局单例，保持不变）
def load_model():
    model = SEResNet50().to(config.DEVICE)
    checkpoint = torch.load(
        config.MODEL_SAVE_PATH,
        map_location=config.DEVICE,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# 初始化模型
model = load_model()

# ---------------------- 1.4.6 版本 GradCAM 初始化（核心修改） ----------------------
# 1. 指定目标层（与你的SEResNet50模型匹配）
target_layer = model.resnet50.layer4[-1]
# 2. 实例化GradCAM（1.4.6 标准写法，无from_config）
gradcam = GradCAM(model=model, target_layers=[target_layer], use_cuda=config.DEVICE == 'cuda')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 上传图片+推理+生成热力图（仅修改热力图生成逻辑）
@upload_bp.route('/', methods=['POST'])
def upload_image():
    try:
        # 1. 接收上传文件（保持不变）
        if 'file' not in request.files:
            return jsonify({'code': 400, 'msg': '未选择文件'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'code': 400, 'msg': '文件名不能为空'})

        # 2. 保存文件（保持不变）
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 3. 预处理图片（新增原始图片numpy格式，用于热力图叠加）
        image = Image.open(file_path).convert('RGB')
        # 原始图片转numpy数组（0-255 → 0-1归一化，1.4.6要求）
        img_np = np.array(image, dtype=np.float32) / 255.0
        # 模型输入张量（保持不变）
        image_tensor = transform(image).to(config.DEVICE)
        input_tensor = image_tensor.unsqueeze(0)  # 增加batch维度 [1,3,224,224]

        # 4. 模型推理（保持不变）
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_label = torch.argmax(outputs, dim=1).item()
            confidence = probs[pred_label]

        # ---------------------- 1.4.6 版本 热力图生成（核心修改） ----------------------
        # 生成灰度热力图（1.4.6 标准调用方式）
        grayscale_cam = gradcam(input_tensor=input_tensor)[0, :]  # 取batch中第一个的热力图
        # 步骤1：获取原始图片的真实尺寸（用户上传的图片可能不是224x224）
        img_height, img_width = img_np.shape[:2]
        # 步骤2：将热力图resize到原始图片尺寸（关键，解决尺寸不一致）
        grayscale_cam = cv2.resize(grayscale_cam, (img_width, img_height))
        # 步骤3：扩展维度，从 (H,W) → (H,W,1)，与 (H,W,3) 的图片兼容
        grayscale_cam = np.expand_dims(grayscale_cam, axis=-1)
        # 叠加热力图到原始图片（1.4.6 专属方法，替代visualize_cam）
        cam_image = show_cam_on_image(
            img_np,  # 原始图片numpy数组（0-1）
            grayscale_cam,  # 灰度热力图
            use_rgb=True  # 原始图片是RGB格式
        )  # 返回值：uint8格式，0-255

        # 6. 保存热力图（保持不变）
        heatmap_filename = f"heatmap_{filename}"
        heatmap_path = os.path.join(config.HEATMAP_FOLDER, heatmap_filename)
        # 转换为BGR格式（cv2保存要求）并保存
        cv2.imwrite(heatmap_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

        # 7. 结果映射 + 保存数据库 + 返回结果（保持不变）解释型用到什么包拿什么
        result = '恶性' if pred_label == 1 else '良性'

        from app import db
        from app.models.record import DetectionRecord
        from datetime import datetime
        record = DetectionRecord(
            filename=filename,
            result=result,
            confidence=float(confidence),
            upload_time=datetime.now()
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            'code': 200,
            'data': {
                'result': result,
                #置信度转换为浮点数后保留4位
                'confidence': round(float(confidence), 4),
                'image_url': f'/static/uploads/{filename}',
                'heatmap_url': f'/static/uploads/heatmaps/{heatmap_filename}',
                'record_id': record.id,
                'upload_time': record.upload_time.strftime('%Y-%m-%d %H:%M:%S')
            }
        })

    except Exception as e:
        return jsonify({'code': 500, 'msg': f'处理失败：{str(e)}'})