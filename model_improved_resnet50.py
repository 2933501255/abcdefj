import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        attn_weights = self.attn(lstm_output)  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # 在序列维度上进行softmax
        
        # 对LSTM输出加权
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output)  # (batch_size, 1, hidden_dim)
        return context.squeeze(1)  # (batch_size, hidden_dim)

class SpatialAttentionWithNorm(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SpatialAttentionWithNorm, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        att_map = self.attention(x)
        return x * att_map

class ImprovedResNet50(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.3, use_layer_norm=True):
        super(ImprovedResNet50, self).__init__()
        
        # 使用ResNet50作为特征提取器 - 保持与原始模型一致
        resnet = models.resnet50(pretrained=True)
        
        # 冻结前几层以减少过拟合
        for param in list(resnet.parameters())[:50]:  # 冻结前几个卷积块
            param.requires_grad = False
            
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        feature_dim = 2048
        
        # 增强的空间注意力 - 带有正则化
        self.spatial_attention = SpatialAttentionWithNorm(feature_dim)
        
        # 特征缩减，降低参数量
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(feature_dim, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True)
        )
        
        # 带有dropout的LSTM - 可配置的dropout率
        self.lstm_dropout = dropout_rate
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # 时序注意力
        self.temporal_attention = TemporalAttention(hidden_dim=512)
        
        # 是否使用LayerNorm
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(512)
        
        # 强化的分类器 - 带有更多的正则化
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),  # 略微减少最后层的dropout
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(-1, c, h, w)  # 合并batch和seq维度以便CNN处理
        
        # 提取CNN特征
        features = self.features(x)
        
        # 应用空间注意力
        features = self.spatial_attention(features)
        
        # 应用特征缩减
        features = self.feature_reduction(features)
        
        # 全局平均池化
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(batch_size, seq_len, -1)
        
        # 应用序列维度的dropout - 提高时序模型的鲁棒性
        if self.training and self.lstm_dropout > 0:
            seq_dropout_mask = torch.ones_like(features[0]) * (1 - self.lstm_dropout)
            seq_dropout_mask = torch.bernoulli(seq_dropout_mask).unsqueeze(0) / (1 - self.lstm_dropout)
            seq_dropout_mask = seq_dropout_mask.expand(batch_size, -1, -1)
            features = features * seq_dropout_mask
        
        # LSTM处理序列特征
        lstm_out, _ = self.lstm(features)
        
        # 应用时序注意力
        temporal_context = self.temporal_attention(lstm_out)
        
        # 应用LayerNorm
        if self.use_layer_norm:
            temporal_context = self.layer_norm(temporal_context)
        
        # 分类
        out = self.classifier(temporal_context)
        
        # 在推断时添加温度缩放，使logits更平滑
        if not self.training:
            out = out / 1.2  # 温度参数 > 1 使分布更平滑
        
        return out

