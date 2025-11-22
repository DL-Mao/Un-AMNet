# ------------------------------------------------------------------
# Barcode Domain Adaptive Discriminator
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttentionModule(nn.Module):
    
    def __init__(self, in_channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B, C, H, W]
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class StripePatternModule(nn.Module):
    
    def __init__(self, in_channels, stripe_kernels=[3, 5, 7]):
        super(StripePatternModule, self).__init__()
        self.stripe_convs = nn.ModuleList()
        
        for kernel_size in stripe_kernels:
            conv_h = nn.Conv2d(in_channels, in_channels // len(stripe_kernels), 
                              kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
            conv_v = nn.Conv2d(in_channels, in_channels // len(stripe_kernels),
                              kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
            
            self.stripe_convs.append(nn.ModuleList([conv_h, conv_v]))
        
        self.fusion = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        stripe_features = []
        
        for conv_h, conv_v in self.stripe_convs:
            h_feat = F.relu(conv_h(x))
            v_feat = F.relu(conv_v(x))
            stripe_features.extend([h_feat, v_feat])
        
        combined = torch.cat(stripe_features, dim=1)
        return self.fusion(combined)


class MultiScaleFeatureProcessor(nn.Module):
    
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super(MultiScaleFeatureProcessor, self).__init__()
        self.scales = scales
        self.processors = nn.ModuleList()
        
        for scale in scales:
            if scale == 1:
                processor = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // len(scales), 3, padding=1),
                    nn.BatchNorm2d(in_channels // len(scales)),
                    nn.ReLU(inplace=True)
                )
            else:
                processor = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // len(scales), 3, padding=1),
                    nn.BatchNorm2d(in_channels // len(scales)),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(scale, scale)
                )
            self.processors.append(processor)
        
        self.fusion = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        multi_scale_features = []
        
        for processor in self.processors:
            feat = processor(x)
            if feat.size()[-2:] != x.size()[-2:]:
                feat = F.interpolate(feat, size=x.size()[-2:], mode='bilinear', align_corners=False)
            multi_scale_features.append(feat)
        
        combined = torch.cat(multi_scale_features, dim=1)
        return self.fusion(combined)


class BarcodeDomainAdaptiveDiscriminator(nn.Module):
    
    def __init__(self, in_planes, n_layers=3, hidden=None, use_spatial_attention=True, 
                 use_stripe_pattern=True, use_multiscale=True):
        super(BarcodeDomainAdaptiveDiscriminator, self).__init__()
        
        self.use_spatial_attention = use_spatial_attention
        self.use_stripe_pattern = use_stripe_pattern
        self.use_multiscale = use_multiscale
        
        _hidden = in_planes if hidden is None else hidden
        
        self.feature_reshape = nn.Linear(in_planes, in_planes)
        
        if use_spatial_attention:
            self.spatial_attention = SpatialAttentionModule(in_planes)
        
        if use_stripe_pattern:
            self.stripe_pattern = StripePatternModule(in_planes)
        
        if use_multiscale:
            self.multiscale_processor = MultiScaleFeatureProcessor(in_planes)
        
        fusion_input_dim = in_planes
        if use_stripe_pattern:
            fusion_input_dim += in_planes
        if use_multiscale:
            fusion_input_dim += in_planes
            
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(fusion_input_dim, _hidden, 1),
            nn.BatchNorm2d(_hidden),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.discriminator_body = nn.Sequential()
        current_dim = _hidden
        
        for i in range(n_layers - 1):
            next_dim = int(current_dim // 1.5) if hidden is None else hidden
            self.discriminator_body.add_module(f'block{i+1}',
                nn.Sequential(
                    nn.Linear(current_dim, next_dim),
                    nn.BatchNorm1d(next_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1)
                ))
            current_dim = next_dim
        
        self.output_layer = nn.Linear(current_dim, 1, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, C]
        Returns:
            output: [B, 1]
        """
        batch_size = x.size(0)
        feature_dim = x.size(1)
        
        target_spatial_size = 32
        
        required_features = target_spatial_size * target_spatial_size
        
        if feature_dim >= required_features:
            x_truncated = x[:, :required_features]
            x_2d = x_truncated.view(batch_size, 1, target_spatial_size, target_spatial_size)
        else:
            repeat_times = (required_features + feature_dim - 1) // feature_dim
            x_repeated = x.repeat(1, repeat_times)
            x_truncated = x_repeated[:, :required_features]
            x_2d = x_truncated.view(batch_size, 1, target_spatial_size, target_spatial_size)
        
        x_2d = x_2d.repeat(1, feature_dim, 1, 1)
        
        if self.use_spatial_attention:
            x_2d = self.spatial_attention(x_2d)
        
        features_to_fuse = [x_2d]
        
        if self.use_stripe_pattern:
            stripe_feat = self.stripe_pattern(x_2d)
            features_to_fuse.append(stripe_feat)
        
        if self.use_multiscale:
            multiscale_feat = self.multiscale_processor(x_2d)
            features_to_fuse.append(multiscale_feat)
        
        fused_features = torch.cat(features_to_fuse, dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        global_features = self.global_pool(fused_features)
        global_features = global_features.view(batch_size, -1)
        
        disc_features = self.discriminator_body(global_features)
        
        output = self.output_layer(disc_features)
        
        return output


class AdaptiveLossFunction(nn.Module):
    
    def __init__(self, margin=0.8, alpha=0.5, beta=0.3):
        super(AdaptiveLossFunction, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, true_scores, fake_scores):
        """
        Args:
            true_scores: discriminator output for real samples
            fake_scores: discriminator output for fake samples
        """
        true_loss = torch.clamp(-true_scores + self.margin, min=0)
        fake_loss = torch.clamp(fake_scores + self.margin, min=0)
        
        true_weight = self.alpha + 0.05 * torch.sigmoid(true_scores)
        fake_weight = self.beta + 0.05 * torch.sigmoid(-fake_scores)
        
        weighted_true_loss = true_loss * true_weight
        weighted_fake_loss = fake_loss * fake_weight
        
        return weighted_true_loss.mean() + weighted_fake_loss.mean()


class BarcodeDiscriminatorWrapper(nn.Module):
    
    def __init__(self, in_planes, n_layers=2, hidden=None, **kwargs):
        super(BarcodeDiscriminatorWrapper, self).__init__()
        
        self.barcode_discriminator = BarcodeDomainAdaptiveDiscriminator(
            in_planes=in_planes,
            n_layers=n_layers,
            hidden=hidden,
            **kwargs
        )
        
        self.adaptive_loss = AdaptiveLossFunction()
    
    def forward(self, x):
        return self.barcode_discriminator(x)
    
    def compute_loss(self, true_scores, fake_scores):
        return self.adaptive_loss(true_scores, fake_scores)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 8
    feature_dim = 1536
    
    print("Testing original discriminator...")
    original_disc = nn.Sequential(
        nn.Linear(feature_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024, 1)
    ).to(device)
    
    test_input = torch.randn(batch_size, feature_dim).to(device)
    original_output = original_disc(test_input)
    print(f"Original discriminator output shape: {original_output.shape}")
    
    print("\nTesting barcode adaptive discriminator...")
    barcode_disc = BarcodeDomainAdaptiveDiscriminator(
        in_planes=feature_dim,
        n_layers=3,
        hidden=1024,
        use_spatial_attention=True,
        use_stripe_pattern=True,
        use_multiscale=True
    ).to(device)
    
    barcode_output = barcode_disc(test_input)
    print(f"Barcode discriminator output shape: {barcode_output.shape}")
    
    print("\nTesting adaptive loss function...")
    true_scores = torch.randn(batch_size, 1).to(device)
    fake_scores = torch.randn(batch_size, 1).to(device)
    
    adaptive_loss = AdaptiveLossFunction()
    loss = adaptive_loss(true_scores, fake_scores)
    print(f"Adaptive loss: {loss.item():.4f}")
    
    print("\nAll tests completed!")
