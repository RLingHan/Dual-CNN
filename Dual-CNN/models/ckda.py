import torch
import torch.nn as nn
import torchvision.models as models


# ==========================
# MSP 模态特有提示（ResNet特征图版本）
# 输入: (B, C, H, W) 的特征图
# ==========================
class MSP_ResNet(nn.Module):
    def __init__(self, in_channels, reduction=4, dropout=0.1):
        super().__init__()
        hid = max(in_channels // reduction, 16)

        # 可见光专用分支
        self.rgb_net = nn.Sequential(
            nn.Conv2d(in_channels, hid, 1, bias=False),   # 1x1卷积压缩通道
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, in_channels, 1, bias=False),   # 恢复通道
        )

        # 红外专用分支
        self.inf_net = nn.Sequential(
            nn.Conv2d(in_channels, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, in_channels, 1, bias=False),
        )

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x, mod):
        """
        x:   (B, C, H, W) 特征图
        mod: (B,) 模态标签，0=红外，1=RGB
        """
        out = torch.zeros_like(x)

        inf_idx = (mod == 0)
        rgb_idx = (mod == 1)

        if inf_idx.sum() > 0:
            out[inf_idx] = self.inf_net(x[inf_idx])
        if rgb_idx.sum() > 0:
            out[rgb_idx] = self.rgb_net(x[rgb_idx])

        return self.dropout(out)


# ==========================
# MCP 模态共有提示（ResNet特征图版本）
# 输入: (B, C, H, W) 的特征图
# ==========================
class MCP_ResNet(nn.Module):
    def __init__(self, in_channels, reduction=8, dropout=0.1):
        super().__init__()
        hid = max(in_channels // reduction, 8)

        # 编码器：压缩到低维提取共有信息
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hid, 1, bias=False),
            nn.BatchNorm2d(hid),
        )

        # 解码器：恢复原始通道数
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, in_channels, 1, bias=False),
        )

        # InstanceNorm去除模态风格，保留结构
        self.IN = nn.InstanceNorm2d(hid, affine=True)

        # 原始特征注意力
        self.att_ori = nn.Sequential(
            nn.Conv2d(hid, max(hid // 4, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(hid // 4, 4), hid, 1, bias=False),
            nn.Sigmoid()
        )

        # 归一化特征注意力
        self.att_in = nn.Sequential(
            nn.Conv2d(hid, max(hid // 4, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(hid // 4, 4), hid, 1, bias=False),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        """
        x: (B, C, H, W) 特征图，RGB和红外混合batch
        """
        x_ori = self.encoder(x)          # 压缩：(B, hid, H, W)
        x_in  = self.IN(x_ori)           # 去模态风格

        att_ori = self.att_ori(x_ori)    # 原始特征权重
        att_in  = self.att_in(x_in)      # 归一化特征权重

        # 自适应融合
        x_merge = att_ori * x_ori + (1 - att_ori) * (att_in * x_in)

        out = self.decoder(x_merge)      # 恢复通道：(B, C, H, W)
        return self.dropout(out)


class ModalPrototypeAlign(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        # 把固定momentum换成可学习的通道级门控
        self.align_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, sub):
        v_feat = x[sub == 0]
        i_feat = x[sub == 1]
        if v_feat.size(0) > 0 and i_feat.size(0) > 0:
            v_proto = v_feat.mean(0, keepdim=True)
            i_proto = i_feat.mean(0, keepdim=True)

            x_new = x.clone()
            # 门控决定每个通道对齐的强度
            gate_v = self.align_gate(x[sub == 0])  # (B_v, C, 1, 1)
            gate_i = self.align_gate(x[sub == 1])  # (B_i, C, 1, 1)

            x_new[sub == 0] = x[sub == 0] + gate_v * (i_proto - v_proto)
            x_new[sub == 1] = x[sub == 1] + gate_i * (v_proto - i_proto)
            return self.proj(x_new)
        return self.proj(x)


# ==========================
# 完整的双模态ResNet50
# ==========================
class DualModalResNet50(nn.Module):
    def __init__(self, num_classes=None, dropout=0.1):
        super().__init__()

        # 加载ResNet50主干
        resnet = models.resnet50(pretrained=True)
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1    # 输出: (B, 256,  64, 32)
        self.layer2  = resnet.layer2    # 输出: (B, 512,  32, 16)
        self.layer3  = resnet.layer3    # 输出: (B, 1024, 16,  8)
        self.layer4  = resnet.layer4    # 输出: (B, 2048,  8,  4)

        # ----------------------------------------
        # 插入点1：layer2之后，通道512
        # 同时加MSP和MCP
        # ----------------------------------------
        self.msp_mid = MSP_ResNet(in_channels=512,  reduction=4, dropout=dropout)
        self.mcp_mid = MCP_ResNet(in_channels=512,  reduction=8, dropout=dropout)

        # ----------------------------------------
        # 插入点2：layer4之后，通道2048
        # 只加MSP，深层做最后的模态特有校正
        # ----------------------------------------
        self.msp_deep = MSP_ResNet(in_channels=2048, reduction=8, dropout=dropout)

        # 模态嵌入标签（对应原论文的mod_embed）
        self.mod_embed_mid  = nn.Parameter(torch.zeros(2, 512,  1, 1))
        self.mod_embed_deep = nn.Parameter(torch.zeros(2, 2048, 1, 1))

        self.gap = nn.AdaptiveAvgPool2d(1)

        if num_classes:
            self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x, mod):
        """
        x:   (B, 3, 256, 128)
        mod: (B,) 0=红外，1=RGB
        """
        # ResNet前处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # ===== 插入点1：layer2之后 =====
        prompt_mid = torch.zeros_like(x)
        prompt_mid = prompt_mid + self.msp_mid(x, mod)   # 模态特有提示
        prompt_mid = prompt_mid + self.mcp_mid(x)         # 模态共有提示

        # 加模态标签嵌入
        inf_idx = (mod == 0)
        rgb_idx = (mod == 1)
        if inf_idx.sum() > 0:
            prompt_mid[inf_idx] += self.mod_embed_mid[0]
        if rgb_idx.sum() > 0:
            prompt_mid[rgb_idx] += self.mod_embed_mid[1]

        x = x + prompt_mid
        # ================================

        x = self.layer3(x)
        x = self.layer4(x)

        # ===== 插入点2：layer4之后 =====
        prompt_deep = self.msp_deep(x, mod)   # 只加模态特有提示

        if inf_idx.sum() > 0:
            prompt_deep[inf_idx] += self.mod_embed_deep[0]
        if rgb_idx.sum() > 0:
            prompt_deep[rgb_idx] += self.mod_embed_deep[1]

        x = x + prompt_deep
        # ================================

        feat = self.gap(x).flatten(1)   # (B, 2048)

        if hasattr(self, 'classifier'):
            return feat, self.classifier(feat)
        return feat


# ==========================
# 快速验证
# ==========================
if __name__ == '__main__':
    model = DualModalResNet50(num_classes=395).cuda()

    # 模拟一个混合batch：4张RGB + 4张红外
    x   = torch.randn(8, 3, 256, 128).cuda()
    mod = torch.tensor([1,1,1,1,0,0,0,0]).cuda()

    feat, logits = model(x, mod)
    print(f"特征维度: {feat.shape}")    # (8, 2048)
    print(f"分类输出: {logits.shape}")  # (8, 395)