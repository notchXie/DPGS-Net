import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet101WithFeatures(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101WithFeatures, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        # 保存ResNet的特征提取部分（不包括最后的全连接层）
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])  # 去掉最后的fc和avgpool
        # 定义一个平均池化层来获取16x16的特征图
        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        # 提取ResNet的中间特征
        feature_map = self.features(x)
        feature_map = self.avgpool(feature_map)
        x = self.resnet.fc(F.adaptive_avg_pool2d(feature_map, (1, 1)).view(feature_map.size(0), -1))

        return x, feature_map


class SEblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEblock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBconvblock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride):
        super(MBconvblock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion

        self.conv1 = nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion)
        self.conv2 = nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride,
                               padding=1, groups=in_channels * expansion, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels * expansion)
        self.conv3 = nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = SEblock(in_channels * expansion, out_channels)

        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.shortcut is not None:
            out += self.shortcut(x)
        return out


# 搭建efficientnet网络
class EfficientNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EfficientNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.mbconv1 = MBconvblock(32, 16, 1, 1)
        self.mbconv2 = MBconvblock(16, 24, 6, 2)
        self.mbconv3 = MBconvblock(24, 40, 6, 2)
        self.mbconv4 = MBconvblock(40, 80, 6, 2)
        self.mbconv5 = MBconvblock(80, 112, 6, 1)
        self.mbconv6 = MBconvblock(112, 192, 6, 2)
        self.mbconv7 = MBconvblock(192, 320, 6, 1)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.mbconv1(out)
        out = self.mbconv2(out)
        out = self.mbconv3(out)
        out = self.mbconv4(out)
        out = self.mbconv5(out)
        out1 = out
        out = self.mbconv6(out)
        out = self.mbconv7(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return torch.softmax(out, dim=1), out1


class EfficientNetWithOneHot(nn.Module):
    def __init__(self, in_channels, out_channels, width_coefficient=1.0, depth_coefficient=1.0):
        super(EfficientNetWithOneHot, self).__init__()
        self.efficientnet = EfficientNet(in_channels, out_channels)

    def forward(self, x):
        out = self.efficientnet(x)  # [batch_size, num_classes]
        print(out)
        class_indices = torch.argmax(out, dim=1)  # [batch_size]
        # 转为独热编码
        one_hot_output = F.one_hot(class_indices, num_classes=out.size(1))  # [batch_size, num_classes]
        one_hot_output = one_hot_output.float()

        return one_hot_output


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DWConv, self).__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias
        )
        # Pointwise Convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)  # Depthwise Convolution
        x = self.pointwise(x)  # Pointwise Convolution
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_radio=16):
        super().__init__()
        self.channels = channels
        self.inter_channels = self.channels  // reduction_radio
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Conv2d(self.channels, self.inter_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.GELU(),
            nn.Conv2d(self.inter_channels, self.channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (b, c, h, w)
        max_out = self.max_pool(x) # (b, c, 1, 1)
        avg_out = self.avg_pool(x) # (b, c, 1, 1)

        max_out = self.mlp(max_out) # (b, c, 1, 1)
        avg_out = self.mlp(avg_out) # (b, c, 1, 1)

        attention = self.sigmoid(max_out + avg_out) #(b, c, 1, 1)

        return attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

class DualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualConvBlock, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.conv9x9 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        #DWConv
        self.dwconv3x3 = nn.Sequential(
            DWConv(out_channels, out_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
        )
        self.dwconv5x5 = nn.Sequential(
            DWConv(out_channels, out_channels*2, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
        )
        self.dwconv7x7 = nn.Sequential(
            DWConv(out_channels, out_channels*2, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(),
        )
        
        # 复合注意力机制
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()

        self.last_conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.last_conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):

        conv1_out = self.conv1x1(x)
        conv5_out = self.conv5x5(x)
        conv9_out = self.conv9x9(x)

        # 通道注意力机制
        conv1_attention = self.channel_attention(conv1_out)
        conv5_attention = self.channel_attention(conv5_out)
        conv9_attention = self.channel_attention(conv9_out)

        conv1_channel = conv1_out * conv1_attention
        conv5_channel = conv5_out * conv5_attention
        conv9_channel = conv9_out * conv9_attention

        out1 = conv1_channel + conv5_channel + conv9_channel

        # 空间注意力机制
        conv1_attention = self.spatial_attention(conv1_out)
        conv5_attention = self.spatial_attention(conv5_out)
        conv9_attention = self.spatial_attention(conv9_out)

        conv1_spatial = conv1_out * conv1_attention
        conv5_spatial = conv5_out * conv5_attention
        conv9_spatial = conv9_out * conv9_attention

        out2 = conv1_spatial + conv5_spatial + conv9_spatial

        out = torch.cat([out1, out2], 1)
        out = self.last_conv2(out)

        out1 = self.dwconv3x3(out)
        out2 = self.dwconv5x5(out)
        out3 = self.dwconv7x7(out)
        out = out1 + out2 + out3
        out = self.last_conv2(out)

        return out


class DualConv3x3_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualConv3x3_block, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            DWConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x

class channel_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(channel_block, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x

class MFDCMmodule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFDCMmodule, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3)

        self.DilatedConv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.DilatedConv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.DilatedConv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=3, stride=1, padding=3, dilation=3, bias=True)

        #DWConv
        self.dwconv3x3 = nn.Sequential(
            DWConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.convFusion = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels*2)

    def forward(self, x):
        output1 = self.gelu(self.bn1(self.conv3x3(x)))
        output2 = self.gelu(self.bn1(self.conv5x5(x)))
        output3 = self.gelu(self.bn1(self.conv7x7(x)))
        output = output1 + output2 + output3

        DilatedConv1 = self.gelu(self.bn2(self.DilatedConv1(output)))
        DilatedConv2 = self.gelu(self.bn2(self.DilatedConv2(output)))
        DilatedConv3 = self.gelu(self.bn2(self.DilatedConv3(output)))
        DilatedConv = DilatedConv1 + DilatedConv2 + DilatedConv3
        DilatedConv = self.convFusion(DilatedConv)

        out = torch.cat([output, DilatedConv], 1)
        out = DilatedConv = self.convFusion(out)
        out = out + output + DilatedConv
        
        return out
        

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.encoder1 = MFDCMmodule(32, 64)
        self.encoder2 = MFDCMmodule(64, 128)
        self.encoder3 = MFDCMmodule(128, 256)
        self.encoder4 = MFDCMmodule(256, 512)

        self.head = DualConvBlock(3, 32)
        self.center1 = MFDCMmodule(512, 1024)
        self.center2 = DualConvBlock(2048, 1024)

        self.decoder4 = DualConv3x3_block(1024, 512)
        self.decoder3 = DualConv3x3_block(512 * 2, 256)
        self.decoder2 = DualConv3x3_block(256 * 2, 128)
        self.decoder1 = DualConv3x3_block(128 * 2, 64)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.up_conv1 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.up_conv2 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.up_conv3 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.up_conv4 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, 1)
        )

        self.resnet = ResNet101WithFeatures(num_classes=4)
        self.channel_block = channel_block(2048, 1024)

    def forward(self, x):
        classifier, classifier_out = self.resnet(x)
        classifier_out = self.channel_block(classifier_out)

        x = self.head(x)  # x: 256*256*64
        x1 = self.encoder1(x)  # x1: 256*256*64
        x2 = self.encoder2(self.max_pool(x1))  # x2: 128*128*128
        x3 = self.encoder3(self.max_pool(x2))  # x3: 64*64*256
        x4 = self.encoder4(self.max_pool(x3))  # x4: 32*32*512

        center = self.center1(self.max_pool(x4))  # center: 16*16*1024
        center = self.center2(torch.cat([center, classifier_out], 1))  # center: 16*16*1024

        decoder4 = self.decoder4(center)  # decoder4: 16*16*512
        decoder4 = self.up_conv1(decoder4)  # decoder4: 32*32*512
        decoder3 = self.decoder3(torch.cat([decoder4, x4], 1))  # decoder3: 32*32*256
        decoder3 = self.up_conv2(decoder3)  # decoder3: 64*64*256
        decoder2 = self.decoder2(torch.cat([decoder3, x3], 1))  # decoder2: 64*64*128
        decoder2 = self.up_conv3(decoder2)  # decoder2: 128*128*128
        decoder1 = self.decoder1(torch.cat([decoder2, x2], 1))  # decoder1: 128*128*64
        decoder1 = self.up_conv4(decoder1)  # decoder1: 256*256*64
        out = torch.cat([decoder1, x1], 1)  # out: 256*256*128

        out = self.final_conv(out)  # out: 256*256*1

        return classifier, out
