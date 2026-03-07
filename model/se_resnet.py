import torch
import torch.nn as nn
import torchvision.models as models
import config


class SEBlock(nn.Module):
    """SE注意力机制模块"""

    def __init__(self, channel, reduction=config.SE_REDUCTION):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResNet50(nn.Module):
    """基于ResNet50的改进模型，在每个残差块后添加SE模块"""

    def __init__(self, num_classes=config.NUM_CLASSES):
        super(SEResNet50, self).__init__()
        # 加载预训练ResNet50
        self.resnet50 = models.resnet50(pretrained=True)
        # 替换最后一层全连接层
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(in_features, num_classes)
        )

        # 在每个残差块后添加SE模块
        self.add_se_modules()

    def add_se_modules(self):
        """遍历ResNet50的层，在每个残差块后插入SE模块"""
        for name, module in self.resnet50.named_children():
            if isinstance(module, nn.Sequential) and 'layer' in name:  # 针对layer1-layer4
                for i, block in enumerate(module):
                    # 残差块的输出通道数 = block.conv3.out_channels
                    se_block = SEBlock(block.conv3.out_channels)
                    # 替换为：原残差块 + SE模块
                    module[i] = nn.Sequential(block, se_block)

    def forward(self, x):
        return self.resnet50(x)


if __name__ == "__main__":
    # 测试模型结构
    model = SEResNet50().to(config.DEVICE)
    x = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
    output = model(x)
    print(f"模型输出形状：{output.shape}")  # 应输出 (1, 2)