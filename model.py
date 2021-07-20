from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import *
from repvgg.repvgg import create_RepVGG_B1

'''
backbone
'''


class ResNet(nn.Module):
    def __init__(self, in_channels=3, layers=34, **kwargs):
        super().__init__()
        supported_layers = {
            18: {'depth': [2, 2, 2, 2], 'block_class': BasicBlock},
            34: {'depth': [3, 4, 6, 3], 'block_class': BasicBlock},
            50: {'depth': [3, 4, 6, 3], 'block_class': BottleneckBlock},
            101: {'depth': [3, 4, 23, 3], 'block_class': BottleneckBlock},
            152: {'depth': [3, 8, 36, 3], 'block_class': BottleneckBlock},
            200: {'depth': [3, 12, 48, 3], 'block_class': BottleneckBlock}
        }
        assert layers in supported_layers, "supported layers are {} but input layer is {}".format(supported_layers,
                                                                                                  layers)

        depth = supported_layers[layers]['depth']
        block_class = supported_layers[layers]['block_class']

        num_filters = [64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            ConvBNACT(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, act='relu'),
            ConvBNACT(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu', padding=1),
            ConvBNACT(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu', padding=1)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList()
        in_ch = 64
        for block_index in range(len(depth)):
            block_list = []
            for i in range(depth[block_index]):
                if layers >= 50:
                    if layers in [101, 152, 200] and block_index == 2:
                        if i == 0:
                            conv_name = "res" + str(block_index + 2) + "a"
                        else:
                            conv_name = "res" + str(block_index + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block_index + 2) + chr(97 + i)
                else:
                    conv_name = f'res{str(block_index + 2)}{chr(97 + i)}'
                if i == 0 and block_index != 0:
                    stride = (2, 1)
                else:
                    stride = (1, 1)
                block_list.append(block_class(in_channels=in_ch, out_channels=num_filters[block_index],
                                              stride=stride,
                                              if_first=block_index == i == 0, name=conv_name))
                in_ch = block_list[-1].output_channels
            self.stages.append(nn.Sequential(*block_list))
        self.out_channels = in_ch
        self.out = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        for stage in self.stages:
            x = stage(x)
        x = self.out(x)
        return x


class MobileNetV3(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.scale = 0.5
        model_name = 'small'
        self.inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', (2, 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (2, 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 1],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', (2, 1)],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', (1, 1)],
                [3, 72, 24, False, 'relu', (2, 1)],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', (2, 1)],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', (2, 1)],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert self.scale in supported_scale, "supported scale are {} but input scale is {}".format(supported_scale,
                                                                                                    self.scale)

        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        cls_ch_squeeze = self.cls_ch_squeeze
        # conv1
        self.conv1 = ConvBNACT(in_channels=in_channels,
                               out_channels=self.make_divisible(inplanes * scale),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               groups=1,
                               act='hard_swish')
        inplanes = self.make_divisible(inplanes * scale)
        block_list = []
        for layer_cfg in cfg:
            block = ResidualUnit(num_in_filter=inplanes,
                                 num_mid_filter=self.make_divisible(scale * layer_cfg[1]),
                                 num_out_filter=self.make_divisible(scale * layer_cfg[2]),
                                 act=layer_cfg[4],
                                 stride=layer_cfg[5],
                                 kernel_size=layer_cfg[0],
                                 use_se=layer_cfg[3])
            block_list.append(block)
            inplanes = self.make_divisible(scale * layer_cfg[2])

        self.blocks = nn.Sequential(*block_list)
        self.conv2 = ConvBNACT(in_channels=inplanes,
                               out_channels=self.make_divisible(scale * cls_ch_squeeze),
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               groups=1,
                               act='hard_swish')

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = self.make_divisible(scale * cls_ch_squeeze)

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x


'''
neck
'''


class SequenceDecoder(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.reshape = Reshape(in_channels)
        self.decoder = DecoderWithRNN(in_channels, **kwargs)
        self.out_channels = self.decoder.out_channels

    def forward(self, x):
        x = self.reshape(x)
        x = self.decoder(x)
        return x


'''
head
'''


class CTC(nn.Module):
    def __init__(self, in_channels, n_class=66, **kwargs):
        super(CTC, self).__init__()
        self.fc = nn.Linear(in_channels, n_class)
        self.n_class = n_class

    def forward(self, x):
        return self.fc(x)


'''
model assemble
through this way we can easily to change the backbone
and with different many combinations 
we can achieve many different results
'''


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.backbone = ResNet()
        self.neck = SequenceDecoder(in_channels=self.backbone.out_channels)
        self.head = CTC(self.neck.out_channels)
        # self.backbone_repvgg = create_RepVGG_B1(deploy=False)

    def forward(self, x):
        # x = self.backbone_repvgg(x)
        # print(self.backbone)
        # print("----------------")
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x
#
# x=torch.randn([1,3,32,120])
# model=CRNN()
# print(model(x))
