import torch
import torch.nn as nn

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

# 多尺度复合注意力特征提取模块
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
        self.center2 = DualConvBlock(1024, 1024)

        self.decoder4 = DualConv3x3_block(1024, 512)
        self.decoder3 = DualConv3x3_block(512*2, 256)
        self.decoder2 = DualConv3x3_block(256*2, 128)
        self.decoder1 = DualConv3x3_block(128*2, 64)


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


    def forward(self, x):

        x = self.head(x)  # x: 256*256*64
        x1 = self.encoder1(x) # x1: 256*256*64
        x2 = self.encoder2(self.max_pool(x1)) # x2: 128*128*128
        x3 = self.encoder3(self.max_pool(x2)) # x3: 64*64*256
        x4 = self.encoder4(self.max_pool(x3)) # x4: 32*32*512

        center = self.center1(self.max_pool(x4)) # center: 16*16*1024
        center = self.center2(center) # center: 16*16*1024

        decoder4 = self.decoder4(center) # decoder4: 16*16*512
        decoder4 = self.up_conv1(decoder4) # decoder4: 32*32*512
        decoder3 = self.decoder3(torch.cat([decoder4, x4], 1)) # decoder3: 32*32*256
        decoder3 = self.up_conv2(decoder3) # decoder3: 64*64*256
        decoder2 = self.decoder2(torch.cat([decoder3, x3], 1)) # decoder2: 64*64*128
        decoder2 = self.up_conv3(decoder2) # decoder2: 128*128*128
        decoder1 = self.decoder1(torch.cat([decoder2, x2], 1)) # decoder1: 128*128*64
        decoder1 = self.up_conv4(decoder1) # decoder1: 256*256*64
        out = torch.cat([decoder1, x1], 1) # out: 256*256*128

        out = self.final_conv(out) # out: 256*256*1

        return out


def test():
    from ptflops import get_model_complexity_info
    x = torch.randn(2, 3, 256, 256)
    # model = DualConvBlock(3, 64)
    # preds = model(x)
    # print(preds.shape)
    #
    # model = DualConv3x3_block(3, 64)
    # preds = model(x)
    # print(preds.shape)
    #
    # model = MFDCMmodule(3, 64)
    # preds = model(x)
    # print(preds.shape)

    model = Unet(3, 1)
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)

    preds = model(x)
    print(preds.shape)

if __name__ == '__main__':
    test()