# ------------------------------------------------------------------
# Simplified Barcode Domain Adaptive Discriminator
# 简化版条码领域自适应判别器
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimplifiedBarcodeDiscriminator(nn.Module):
    """简化版条码自适应判别器 - 更稳定的实现"""
    
    def __init__(self, in_planes, n_layers=3, hidden=None):
        super(SimplifiedBarcodeDiscriminator, self).__init__()
        
        _hidden = in_planes if hidden is None else hidden
        
        # 特征投影层 - 将高维特征投影到固定维度
        self.feature_projection = nn.Sequential(
            nn.Linear(in_planes, _hidden),
            nn.BatchNorm1d(_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 空间特征提取 - 使用1D卷积模拟空间处理
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(_hidden, _hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(_hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(_hidden, _hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(_hidden),
            nn.ReLU(inplace=True)
        )
        
        # 条纹模式检测 - 使用不同尺度的卷积核
        self.stripe_detector = nn.ModuleList([
            nn.Conv1d(_hidden, _hidden // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(_hidden * 2, _hidden),
            nn.BatchNorm1d(_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 判别器主体
        self.discriminator_body = nn.Sequential()
        current_dim = _hidden
        
        for i in range(n_layers - 1):
            next_dim = int(current_dim // 1.5)
            self.discriminator_body.add_module(f'block{i+1}',
                nn.Sequential(
                    nn.Linear(current_dim, next_dim),
                    nn.BatchNorm1d(next_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1)
                ))
            current_dim = next_dim
        
        # 输出层
        self.output_layer = nn.Linear(current_dim, 1, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, C] - 输入特征
        Returns:
            output: [B, 1] - 判别器输出
        """
        batch_size = x.size(0)
        
        # 特征投影
        projected_features = self.feature_projection(x)  # [B, hidden]
        
        # 空间特征提取
        spatial_features = projected_features.unsqueeze(2)  # [B, hidden, 1]
        spatial_features = self.spatial_conv(spatial_features)  # [B, hidden, 1]
        spatial_features = spatial_features.squeeze(2)  # [B, hidden]
        
        # 条纹模式检测
        stripe_input = projected_features.unsqueeze(2)  # [B, hidden, 1]
        stripe_features = []
        for stripe_conv in self.stripe_detector:
            stripe_feat = F.relu(stripe_conv(stripe_input))  # [B, hidden//4, 1]
            stripe_features.append(stripe_feat.squeeze(2))  # [B, hidden//4]
        
        # 融合条纹特征
        stripe_combined = torch.cat(stripe_features, dim=1)  # [B, hidden]
        
        # 特征融合
        combined_features = torch.cat([spatial_features, stripe_combined], dim=1)  # [B, hidden*2]
        fused_features = self.feature_fusion(combined_features)  # [B, hidden]
        
        # 判别器主体
        disc_features = self.discriminator_body(fused_features)
        
        # 输出
        output = self.output_layer(disc_features)
        
        return output


class AdaptiveLossFunction(nn.Module):
    """自适应损失函数 - 针对条码检测优化"""
    
    def __init__(self, margin=0.8, alpha=0.8, beta=0.6):
        super(AdaptiveLossFunction, self).__init__()
        self.margin = margin
        self.alpha = alpha  # 真实样本权重

        self.beta = beta     # 假样本权重
        
    def forward(self, true_scores, fake_scores):
        """
        Args:
            true_scores: 真实样本的判别器输出
            fake_scores: 假样本的判别器输出
        """
        # 基础margin loss
        true_loss = torch.clamp(-true_scores + self.margin, min=0)
        fake_loss = torch.clamp(fake_scores + self.margin, min=0)
        
        # 自适应权重
        true_weight = self.alpha + 0.9 * torch.sigmoid(true_scores)
        fake_weight = self.beta + 0.9 * torch.sigmoid(-fake_scores)
        
        # 加权损失
        weighted_true_loss = true_loss * true_weight
        weighted_fake_loss = fake_loss * fake_weight
        
        return weighted_true_loss.mean() + weighted_fake_loss.mean()


# 兼容性包装器
class BarcodeDiscriminatorWrapper(nn.Module):
    """条码判别器包装器 - 保持与原始判别器的接口兼容"""
    
    def __init__(self, in_planes, n_layers=2, hidden=None, **kwargs):
        super(BarcodeDiscriminatorWrapper, self).__init__()
        
        # 创建简化版条码自适应判别器
        self.barcode_discriminator = SimplifiedBarcodeDiscriminator(
            in_planes=in_planes,
            n_layers=n_layers,
            hidden=hidden
        )
        
        # 创建自适应损失函数
        self.adaptive_loss = AdaptiveLossFunction()
    
    def forward(self, x):
        return self.barcode_discriminator(x)
    
    def compute_loss(self, true_scores, fake_scores):
        """计算自适应损失"""
        return self.adaptive_loss(true_scores, fake_scores)


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 8
    feature_dim = 1536
    
    print("Testing Simplified Barcode Discriminator...")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 测试简化版判别器
    discriminator = SimplifiedBarcodeDiscriminator(
        in_planes=feature_dim,
        n_layers=3,
        hidden=1024
    ).to(device)
    
    test_input = torch.randn(batch_size, feature_dim).to(device)
    output = discriminator(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # 测试损失函数
    true_scores = torch.randn(batch_size, 1).to(device)
    fake_scores = torch.randn(batch_size, 1).to(device)
    
    adaptive_loss = AdaptiveLossFunction()
    loss = adaptive_loss(true_scores, fake_scores)
    print(f"Loss value: {loss.item():.4f}")
    
    print("All tests passed!")
