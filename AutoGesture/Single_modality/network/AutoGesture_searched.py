import torch
import torch.nn as nn
import torch.nn.functional as F
from network.operations import *
# from operations import *

from torch.autograd import Variable
import pdb


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction_prev, normal):
        super(Cell, self).__init__()

        # stride = [1,2,2]
        if reduction_prev:
            self.preprocess0 = VaniConv3d(C_prev_prev, C, 1, [1, 2, 2], 0, affine=False)
        else:
            self.preprocess0 = VaniConv3d(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = VaniConv3d(C_prev, C, 1, 1, 0, affine=False)

        # branch 8 frame
        if normal == 1:
            op_names, indices = zip(*genotype.normal8)
            concat = genotype.normal_concat8
        # branch 16 frame
        if normal == 2:
            op_names, indices = zip(*genotype.normal16)
            concat = genotype.normal_concat16
        # branch 32 frame
        if normal == 3:
            op_names, indices = zip(*genotype.normal32)
            concat = genotype.normal_concat32

        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        # pdb.set_trace()

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


# steps = how many nodes in a cell; multiplier = how many nodes to be concated
# Default, layer = 8, steps = 4, multiplier = 4
# [3 x 16 x 112 x 112]
# 8 layers, each 2 layers with one MaxPool(stride=[1,2,2])
class AutoGesture(nn.Module):
    def __init__(self, C8, C16, C32, num_classes, layers, genotype):
        super(AutoGesture, self).__init__()
        self._C8 = C8  # 48
        self._C16 = C16  # 32
        self._C32 = C32  # 16
        self._num_classes = num_classes
        self._layers = layers

        self.MaxpoolSpa = nn.MaxPool3d(kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=[1, 2, 2])
        self.AvgpoolSpa = nn.AvgPool3d(kernel_size=2, stride=2)

        # lateral 16
        self.lateral16_1 = nn.Sequential(
            nn.Conv3d(C16 * 2, C16 * 2, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C16 * 2)
        )
        self.lateral16_2 = nn.Sequential(
            nn.Conv3d(C16 * 2 * 4, C16 * 2, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C16 * 2)
        )
        self.lateral16_3 = nn.Sequential(
            nn.Conv3d(C16 * 4 * 4, C16 * 4, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C16 * 4)
        )

        # lateral 32
        self.lateral32_16_1 = nn.Sequential(
            nn.Conv3d(C32 * 2, C32 * 2, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 2)
        )
        self.lateral32_16_2 = nn.Sequential(
            nn.Conv3d(C32 * 2 * 4, C32 * 2, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 2)
        )
        self.lateral32_16_3 = nn.Sequential(
            nn.Conv3d(C32 * 4 * 4, C32 * 4, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 4)
        )
        self.lateral32_8_1 = nn.Sequential(
            nn.Conv3d(C32 * 2, C32 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 2)
        )
        self.lateral32_8_2 = nn.Sequential(
            nn.Conv3d(C32 * 2 * 4, C32 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 2)
        )
        self.lateral32_8_3 = nn.Sequential(
            nn.Conv3d(C32 * 4 * 4, C32 * 4, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 4)
        )

        # 8 frames
        self.stem0_8 = nn.Sequential(
            nn.Conv3d(3, C8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(C8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=[1, 2, 2]),
        )
        self.stem1_8 = nn.Sequential(

            nn.Conv3d(C8, 2 * C8, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(2 * C8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, padding=1, stride=2),
        )

        # 16 frames
        self.stem0_16 = nn.Sequential(
            nn.Conv3d(3, C16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(C16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=[1, 2, 2]),
        )
        self.stem1_16 = nn.Sequential(

            nn.Conv3d(C16, 2 * C16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(2 * C16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, padding=1, stride=2),
        )

        # 32 frames
        self.stem0_32 = nn.Sequential(
            nn.Conv3d(3, C32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(C32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=[1, 2, 2]),
        )
        self.stem1_32 = nn.Sequential(

            nn.Conv3d(C32, 2 * C32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(2 * C32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, padding=1, stride=2),
        )

        C_prev_prev8, C_prev8, C_curr8 = C8, (2 * C8 + 2 * C16 + 2 * C32), 2 * C8
        C_prev_prev16, C_prev16, C_curr16 = C16, (2 * C16 + 2 * C32), 2 * C16
        C_prev_prev32, C_prev32, C_curr32 = C32, 2 * C32, 2 * C32

        # 8
        self.cells8 = nn.ModuleList()
        normal = 1
        reduction_prev = False
        for i in range(layers):
            if i in [3, 6]:
                C_curr8 *= 2
                reduction_prev = True

                if i == 3:
                    C_prev8 = 2 * C8 * 4 + 2 * C16 + 2 * C32

                if i == 6:
                    C_prev8 = 4 * C8 * 4 + 4 * C16 + 4 * C32

            else:
                reduction_prev = False
            cell = Cell(genotype, C_prev_prev8, C_prev8, C_curr8, reduction_prev, normal)
            self.cells8 += [cell]
            C_prev_prev8, C_prev8 = C_prev8, cell.multiplier * C_curr8

        # 16
        self.cells16 = nn.ModuleList()
        normal = 2
        reduction_prev = False
        for i in range(layers):
            if i in [3, 6]:
                C_curr16 *= 2
                reduction_prev = True

                if i == 3:
                    C_prev16 = 2 * C16 * 4 + 2 * C32

                if i == 6:
                    C_prev16 = 4 * C16 * 4 + 4 * C32
            else:
                reduction_prev = False
            cell = Cell(genotype, C_prev_prev16, C_prev16, C_curr16, reduction_prev, normal)
            self.cells16 += [cell]
            C_prev_prev16, C_prev16 = C_prev16, cell.multiplier * C_curr16

        # 32
        self.cells32 = nn.ModuleList()
        normal = 3
        reduction_prev = False
        for i in range(layers):
            if i in [3, 6]:
                C_curr32 *= 2
                reduction_prev = True
            else:
                reduction_prev = False
            cell = Cell(genotype, C_prev_prev32, C_prev32, C_curr32, reduction_prev, normal)
            self.cells32 += [cell]
            C_prev_prev32, C_prev32 = C_prev32, cell.multiplier * C_curr32

        # head
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(C_prev8 + C_prev16 + C_prev32, num_classes)

    def forward(self, inputs):

        ## inputs 32 x 3 x 112 x 112 ##

        # 32
        s0_32 = self.stem0_32(inputs)  # 32*28*28
        s1_32 = self.stem1_32(s0_32)  # 16*28*28
        s0_32 = self.AvgpoolSpa(s0_32)  # 16*28*28
        s1_32_lateral_16_1 = self.lateral32_16_1(s1_32)
        s1_32_lateral_8_1 = self.lateral32_8_1(s1_32)

        # 16
        s0_16 = self.stem0_16(inputs[:, :, ::2, :, :])  # 16*28*28
        s1_16 = self.stem1_16(s0_16)  # 8*28*28
        s0_16 = self.AvgpoolSpa(s0_16)  # 8*28*28
        s1_16_lateral1 = self.lateral16_1(s1_16)

        s1_16 = torch.cat([s1_32_lateral_16_1, s1_16], dim=1)

        # 8
        s0_8 = self.stem0_8(inputs[:, :, ::4, :, :])  # 8*28*28
        s1_8 = self.stem1_8(s0_8)  # 4*28*28
        s0_8 = self.AvgpoolSpa(s0_8)  # 4*28*28

        s1_8 = torch.cat([s1_32_lateral_8_1, s1_16_lateral1, s1_8], dim=1)

        # 32
        for ii, cell in enumerate(self.cells32):

            s0_32, s1_32 = s1_32, cell(s0_32, s1_32)

            if ii == 2:
                s1_32 = self.MaxpoolSpa(s1_32)
                s1_32_lateral_16_2 = self.lateral32_16_2(s1_32)
                s1_32_lateral_8_2 = self.lateral32_8_2(s1_32)

            if ii == 5:
                s1_32 = self.MaxpoolSpa(s1_32)
                s1_32_lateral_16_3 = self.lateral32_16_3(s1_32)
                s1_32_lateral_8_3 = self.lateral32_8_3(s1_32)

                # 16
        for ii, cell in enumerate(self.cells16):

            s0_16, s1_16 = s1_16, cell(s0_16, s1_16)

            if ii == 2:
                branch16_Feature28 = s1_16  # [64, 8, 28, 28]
                s1_16 = self.MaxpoolSpa(s1_16)
                s1_16_lateral2 = self.lateral16_2(s1_16)
                s1_16 = torch.cat([s1_32_lateral_16_2, s1_16], dim=1)

            if ii == 5:
                branch16_Feature14 = s1_16  # [128, 8, 14, 14]
                s1_16 = self.MaxpoolSpa(s1_16)
                s1_16_lateral3 = self.lateral16_3(s1_16)
                s1_16 = torch.cat([s1_32_lateral_16_3, s1_16], dim=1)

            if ii == 7:
                branch16_Feature7 = s1_16  # [128, 8, 7, 7]

        # 8
        for ii, cell in enumerate(self.cells8):

            s0_8, s1_8 = s1_8, cell(s0_8, s1_8)

            if ii == 2:
                s1_8 = self.MaxpoolSpa(s1_8)
                s1_8 = torch.cat([s1_32_lateral_8_2, s1_16_lateral2, s1_8], dim=1)

            if ii == 5:
                s1_8 = self.MaxpoolSpa(s1_8)
                s1_8 = torch.cat([s1_32_lateral_8_3, s1_16_lateral3, s1_8], dim=1)

        out_32 = self.global_pooling(s1_32)  # C32*8*4*=8*8*4=512
        out_16 = self.global_pooling(s1_16)  # C16*8*4*=16*8*4=1024
        out_8 = self.global_pooling(s1_8)  # C8*8*4=24*8*4=1592
        out = torch.cat([out_32, out_16, out_8], dim=1)
        logits = self.classifier(self.dropout(out.view(out.size(0), -1)))  # dropout=0.5
        # logits = self.classifier(out.view(out.size(0),-1))

        # return logits, branch16_Feature28, branch16_Feature14, branch16_Feature7
        return logits


# steps = how many nodes in a cell; multiplier = how many nodes to be concated
# Default, layer = 12, steps = 4, multiplier = 4
# [3 x 16 x 112 x 112]
# 8 layers, each 2 layers with one MaxPool(stride=[1,2,2])
class AutoGesture_12layers(nn.Module):
    def __init__(self, C8, C16, C32, num_classes, layers, genotype):
        super(AutoGesture_12layers, self).__init__()
        self._C8 = C8  # 48
        self._C16 = C16  # 32
        self._C32 = C32  # 16
        self._num_classes = num_classes
        self._layers = layers

        self.MaxpoolSpa = nn.MaxPool3d(kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=[1, 2, 2])
        self.AvgpoolSpa = nn.AvgPool3d(kernel_size=2, stride=2)

        # lateral 16
        self.lateral16_1 = nn.Sequential(
            nn.Conv3d(C16 * 2, C16 * 2, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C16 * 2)
        )
        self.lateral16_2 = nn.Sequential(
            nn.Conv3d(C16 * 2 * 4, C16 * 2, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C16 * 2)
        )
        self.lateral16_3 = nn.Sequential(
            nn.Conv3d(C16 * 4 * 4, C16 * 4, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C16 * 4)
        )

        # lateral 32
        self.lateral32_16_1 = nn.Sequential(
            nn.Conv3d(C32 * 2, C32 * 2, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 2)
        )
        self.lateral32_16_2 = nn.Sequential(
            nn.Conv3d(C32 * 2 * 4, C32 * 2, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 2)
        )
        self.lateral32_16_3 = nn.Sequential(
            nn.Conv3d(C32 * 4 * 4, C32 * 4, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 4)
        )
        self.lateral32_8_1 = nn.Sequential(
            nn.Conv3d(C32 * 2, C32 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 2)
        )
        self.lateral32_8_2 = nn.Sequential(
            nn.Conv3d(C32 * 2 * 4, C32 * 2, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 2)
        )
        self.lateral32_8_3 = nn.Sequential(
            nn.Conv3d(C32 * 4 * 4, C32 * 4, kernel_size=(5, 1, 1), stride=(4, 1, 1), bias=False, padding=(2, 0, 0)),
            nn.BatchNorm3d(C32 * 4)
        )

        # 8 frames
        self.stem0_8 = nn.Sequential(
            nn.Conv3d(3, C8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(C8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=[1, 2, 2]),
        )
        self.stem1_8 = nn.Sequential(

            nn.Conv3d(C8, 2 * C8, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(2 * C8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, padding=1, stride=2),
        )

        # 16 frames
        self.stem0_16 = nn.Sequential(
            nn.Conv3d(3, C16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(C16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=[1, 2, 2]),
        )
        self.stem1_16 = nn.Sequential(

            nn.Conv3d(C16, 2 * C16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(2 * C16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, padding=1, stride=2),
        )

        # 32 frames
        self.stem0_32 = nn.Sequential(
            nn.Conv3d(3, C32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(C32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=[1, 2, 2]),
        )
        self.stem1_32 = nn.Sequential(

            nn.Conv3d(C32, 2 * C32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(2 * C32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, padding=1, stride=2),
        )

        C_prev_prev8, C_prev8, C_curr8 = C8, (2 * C8 + 2 * C16 + 2 * C32), 2 * C8
        C_prev_prev16, C_prev16, C_curr16 = C16, (2 * C16 + 2 * C32), 2 * C16
        C_prev_prev32, C_prev32, C_curr32 = C32, 2 * C32, 2 * C32

        # 8
        self.cells8 = nn.ModuleList()
        normal = 1
        reduction_prev = False
        for i in range(layers):
            if i in [4, 9]:
                C_curr8 *= 2
                reduction_prev = True

                if i == 4:
                    C_prev8 = 2 * C8 * 4 + 2 * C16 + 2 * C32

                if i == 9:
                    C_prev8 = 4 * C8 * 4 + 4 * C16 + 4 * C32

            else:
                reduction_prev = False
            cell = Cell(genotype, C_prev_prev8, C_prev8, C_curr8, reduction_prev, normal)
            self.cells8 += [cell]
            C_prev_prev8, C_prev8 = C_prev8, cell.multiplier * C_curr8

        # 16
        self.cells16 = nn.ModuleList()
        normal = 2
        reduction_prev = False
        for i in range(layers):
            if i in [4, 9]:
                C_curr16 *= 2
                reduction_prev = True

                if i == 4:
                    C_prev16 = 2 * C16 * 4 + 2 * C32

                if i == 9:
                    C_prev16 = 4 * C16 * 4 + 4 * C32
            else:
                reduction_prev = False
            cell = Cell(genotype, C_prev_prev16, C_prev16, C_curr16, reduction_prev, normal)
            self.cells16 += [cell]
            C_prev_prev16, C_prev16 = C_prev16, cell.multiplier * C_curr16

        # 32
        self.cells32 = nn.ModuleList()
        normal = 3
        reduction_prev = False
        for i in range(layers):
            if i in [4, 9]:
                C_curr32 *= 2
                reduction_prev = True
            else:
                reduction_prev = False
            cell = Cell(genotype, C_prev_prev32, C_prev32, C_curr32, reduction_prev, normal)
            self.cells32 += [cell]
            C_prev_prev32, C_prev32 = C_prev32, cell.multiplier * C_curr32

        # head
        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(C_prev8 + C_prev16 + C_prev32, num_classes)

    def forward(self, inputs):

        ## inputs 32 x 3 x 112 x 112 ##

        # 32
        s0_32 = self.stem0_32(inputs)  # 32*28*28
        s1_32 = self.stem1_32(s0_32)  # 16*28*28
        s0_32 = self.AvgpoolSpa(s0_32)  # 16*28*28
        s1_32_lateral_16_1 = self.lateral32_16_1(s1_32)
        s1_32_lateral_8_1 = self.lateral32_8_1(s1_32)

        # 16
        s0_16 = self.stem0_16(inputs[:, :, ::2, :, :])  # 16*28*28
        s1_16 = self.stem1_16(s0_16)  # 8*28*28
        s0_16 = self.AvgpoolSpa(s0_16)  # 8*28*28
        s1_16_lateral1 = self.lateral16_1(s1_16)

        s1_16 = torch.cat([s1_32_lateral_16_1, s1_16], dim=1)

        # 8
        s0_8 = self.stem0_8(inputs[:, :, ::4, :, :])  # 8*28*28
        s1_8 = self.stem1_8(s0_8)  # 4*28*28
        s0_8 = self.AvgpoolSpa(s0_8)  # 4*28*28

        s1_8 = torch.cat([s1_32_lateral_8_1, s1_16_lateral1, s1_8], dim=1)

        # 32
        for ii, cell in enumerate(self.cells32):

            s0_32, s1_32 = s1_32, cell(s0_32, s1_32)

            if ii == 3:
                s1_32 = self.MaxpoolSpa(s1_32)
                s1_32_lateral_16_2 = self.lateral32_16_2(s1_32)
                s1_32_lateral_8_2 = self.lateral32_8_2(s1_32)

            if ii == 8:
                s1_32 = self.MaxpoolSpa(s1_32)
                s1_32_lateral_16_3 = self.lateral32_16_3(s1_32)
                s1_32_lateral_8_3 = self.lateral32_8_3(s1_32)

                # 16
        for ii, cell in enumerate(self.cells16):

            s0_16, s1_16 = s1_16, cell(s0_16, s1_16)

            if ii == 3:
                branch16_Feature28 = s1_16  # [64, 8, 28, 28]
                s1_16 = self.MaxpoolSpa(s1_16)
                s1_16_lateral2 = self.lateral16_2(s1_16)
                s1_16 = torch.cat([s1_32_lateral_16_2, s1_16], dim=1)

            if ii == 8:
                branch16_Feature14 = s1_16  # [128, 8, 14, 14]
                s1_16 = self.MaxpoolSpa(s1_16)
                s1_16_lateral3 = self.lateral16_3(s1_16)
                s1_16 = torch.cat([s1_32_lateral_16_3, s1_16], dim=1)

            if ii == 11:
                branch16_Feature7 = s1_16  # [128, 8, 7, 7]

        # 8
        for ii, cell in enumerate(self.cells8):

            s0_8, s1_8 = s1_8, cell(s0_8, s1_8)

            if ii == 3:
                s1_8 = self.MaxpoolSpa(s1_8)
                s1_8 = torch.cat([s1_32_lateral_8_2, s1_16_lateral2, s1_8], dim=1)

            if ii == 8:
                s1_8 = self.MaxpoolSpa(s1_8)
                s1_8 = torch.cat([s1_32_lateral_8_3, s1_16_lateral3, s1_8], dim=1)

        out_32 = self.global_pooling(s1_32)  # C32*8*4*=8*8*4=512
        out_16 = self.global_pooling(s1_16)  # C16*8*4*=16*8*4=1024
        out_8 = self.global_pooling(s1_8)  # C8*8*4=24*8*4=1592
        out = torch.cat([out_32, out_16, out_8], dim=1)
        logits = self.classifier(self.dropout(out.view(out.size(0), -1)))  # dropout=0.5
        # logits = self.classifier(out.view(out.size(0),-1))

        # return logits, branch16_Feature28, branch16_Feature14, branch16_Feature7
        return logits
if __name__ == '__main__':
    from collections import namedtuple
    import os

    Genotype_Net = namedtuple('Genotype', 'normal8 normal_concat8 normal16 normal_concat16 normal32 normal_concat32')

    PRIMITIVES = [
        'none',
        'skip_connect',
        'TCDC06_3x3x3',
        'TCDC03avg_3x3x3',
        'conv_1x3x3',
        'TCDC06_3x1x1',
        'TCDC03avg_3x1x1',
    ]

    # AutoGesture_3Branch_CDC_RGB_New
    genotype_RGB = Genotype_Net(
        normal8=[('TCDC06_3x1x1', 1), ('TCDC06_3x3x3', 0), ('skip_connect', 0), ('skip_connect', 1),
                 ('skip_connect', 0), ('skip_connect', 1), ('TCDC06_3x1x1', 2), ('skip_connect', 3)],
        normal_concat8=range(2, 6),
        normal16=[('TCDC06_3x1x1', 1), ('skip_connect', 0), ('TCDC06_3x1x1', 1), ('TCDC06_3x3x3', 2),
                  ('TCDC06_3x3x3', 3), ('TCDC06_3x1x1', 1), ('TCDC03avg_3x3x3', 2), ('TCDC06_3x3x3', 3)],
        normal_concat16=range(2, 6),
        normal32=[('TCDC03avg_3x3x3', 1), ('skip_connect', 0), ('conv_1x3x3', 1), ('conv_1x3x3', 0),
                  ('TCDC06_3x1x1', 1), ('skip_connect', 2), ('TCDC06_3x1x1', 1), ('TCDC06_3x3x3', 0)],
        normal_concat32=range(2, 6))

    init_channels8 = 16
    init_channels16 = 16
    init_channels32 = 16
    classes = 249
    layers = 12
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    model = AutoGesture_12layers(init_channels8, init_channels16, init_channels32, classes, layers, genotype_RGB)
    print(model)
    input()
    inputs = torch.randn(2, 3, 32, 112, 112)
    # outputs = model(inputs)
    logits = model(inputs)
    print(logits.shape)