import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'skip_connect' : lambda C, stride, affine: Identity(),
  'conv_3x3x3' : lambda C, stride, affine: VaniConv3d(C, C, 3, stride, 1, affine=affine),
  'STCDC06_3x3x3' : lambda C, stride, affine: CDC_Unit(C, C, 3, stride, 1, basic_conv=ST_3DCDC, theta=0.6, affine=affine),
  'TCDC06_3x3x3' : lambda C, stride, affine: CDC_Unit(C, C, 3, stride, 1, basic_conv=T_3DCDC, theta=0.6, affine=affine),
  'TCDC03avg_3x3x3' : lambda C, stride, affine: CDC_Unit(C, C, 3, stride, 1, basic_conv=T_3DCDC_Avg, theta=0.3, affine=affine),
  'TCDC06_3x1x1' : lambda C, stride, affine: CDC_Unit(C, C, 3, stride, padding=1, basic_conv=T_3DCDC, theta=0.6, affine=affine),
  'TCDC03avg_3x1x1' : lambda C, stride, affine: CDC_Unit(C, C, 3, stride, padding=1, basic_conv=T_3DCDC_Avg, theta=0.3, affine=affine),
  'dil_3x3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, padding=2, dilation=2,affine=affine),
  'STCDC06_dil_3x3x3' : lambda C, stride, affine: Dil_STCDC(C, C, 3, stride, padding=2, dilation=2, theta=0.6, affine=affine),
  'TCDC06_dil_3x3x3' : lambda C, stride, affine: Dil_TCDC(C, C, 3, stride, padding=2, dilation=2, theta=0.6, affine=affine),
  'conv_1x3x3' : lambda C, stride, affine: VaniConv3d_Spatial_1x3x3(C, C, 3, stride, 1, affine=affine),
  'conv_3x1x1' : lambda C, stride, affine: VaniConv3d_Temporal_3x1x1(C, C, 3, stride, 1, affine=affine),
  
  # stride = 2
  'MaxPool_3x1x1' : lambda C, stride, affine: MaxPool_3x1x1(C, C, 3, stride, 1, affine=affine),
  'AvgPool_3x1x1' : lambda C, stride, affine: AvgPool_3x1x1(C, C, 3, stride, 1, affine=affine),
  # stride = 4
  'TCDC06_5x1x1' : lambda C, stride, affine: CDC_Unit(C, C, 5, stride, padding=2, basic_conv=T_3DCDC, theta=0.6, affine=affine),
  'TCDC03avg_5x1x1' : lambda C, stride, affine: CDC_Unit(C, C, 5, stride, padding=2, basic_conv=T_3DCDC_Avg, theta=0.3, affine=affine),
  'conv_5x1x1' : lambda C, stride, affine: VaniConv3d_Temporal_5x1x1(C, C, 5, stride, 2, affine=affine),
  'MaxPool_5x1x1' : lambda C, stride, affine: MaxPool_5x1x1(stride),
  'AvgPool_5x1x1' : lambda C, stride, affine: AvgPool_5x1x1(stride),
 
}


OPS_connection = {
  'none' : lambda C_in, C_out, stride, affine: Zero_Connection(C_in, C_out, stride), 
  # stride = 2
  'TCDC06_3x1x1' : lambda C_in, C_out, stride, affine: CDC_Unit(C_in, C_out, 3, stride, padding=1, basic_conv=T_3DCDC, theta=0.6, affine=affine),
  'TCDC03avg_3x1x1' : lambda C_in, C_out, stride, affine: CDC_Unit(C_in, C_out, 3, stride, padding=1, basic_conv=T_3DCDC_Avg, theta=0.3, affine=affine),
  'MaxPool_3x1x1' : lambda C_in, C_out, stride, affine: MaxPool_3x1x1(C_in, C_out, stride, affine=affine),
  'AvgPool_3x1x1' : lambda C_in, C_out, stride, affine: AvgPool_3x1x1(C_in, C_out, stride, affine=affine),
  'conv_3x1x1' : lambda C_in, C_out, stride, affine: VaniConv3d_Temporal_3x1x1(C_in, C_out, 3, stride, 1, affine=affine),
  # stride = 4
  'TCDC06_5x1x1' : lambda C_in, C_out, stride, affine: CDC_Unit(C_in, C_out, 5, stride, padding=2, basic_conv=T_3DCDC, theta=0.6, affine=affine),
  'TCDC03avg_5x1x1' : lambda C_in, C_out, stride, affine: CDC_Unit(C_in, C_out, 5, stride, padding=2, basic_conv=T_3DCDC_Avg, theta=0.3, affine=affine),
  'conv_5x1x1' : lambda C_in, C_out, stride, affine: VaniConv3d_Temporal_5x1x1(C_in, C_out, 5, stride, 2, affine=affine),
  'MaxPool_5x1x1' : lambda C_in, C_out, stride, affine: MaxPool_5x1x1(C_in, C_out, stride, affine=affine),
  'AvgPool_5x1x1' : lambda C_in, C_out, stride, affine: AvgPool_5x1x1(C_in, C_out, stride, affine=affine),
 
}





class ST_3DCDC(nn.Module):
    '''
    Spatio-Temporal Center-difference based Convolutional layer (3D version) 
    theta: control the percentage of original convolution and centeral-difference convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(ST_3DCDC, self).__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in,t,kernel_size,kernel_size] = self.conv.weight.shape
            
            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2]>1:
                kernel_diff = self.conv.weight.sum(2).sum(2).sum(2) 
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
                return out_normal - self.theta * out_diff
                
            else:
                return out_normal







class T_3DCDC(nn.Module):
    '''
    Temporal Center-difference based Convolutional layer (3D version) 
    theta: control the percentage of original convolution and centeral-difference convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(T_3DCDC, self).__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            
            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2]>1:
                if self.conv.weight.shape[2]==3:
                    kernel_diff = self.conv.weight[:,:,0,:,:].sum(2).sum(2) + self.conv.weight[:,:,2,:,:].sum(2).sum(2)
                if self.conv.weight.shape[2]==5:
                    kernel_diff = self.conv.weight[:,:,0,:,:].sum(2).sum(2) + self.conv.weight[:,:,1,:,:].sum(2).sum(2) + self.conv.weight[:,:,3,:,:].sum(2).sum(2) + self.conv.weight[:,:,4,:,:].sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
                return out_normal - self.theta * out_diff
                
            else:
                return out_normal




class T_3DCDC_Avg(nn.Module):
    '''
    Temporal Center-difference based Convolutional layer (3D version) 
    theta: control the percentage of original convolution and centeral-difference convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(T_3DCDC_Avg, self).__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.avgpool = nn.AvgPool3d(kernel_size=(kernel_size,1,1), stride=stride, padding=(padding,0,0))
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        local_avg = self.avgpool(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            
            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2]>1:
                if self.conv.weight.shape[2]==3:
                    kernel_diff = self.conv.weight[:,:,0,:,:].sum(2).sum(2) + self.conv.weight[:,:,2,:,:].sum(2).sum(2)
                if self.conv.weight.shape[2]==5:
                    kernel_diff = self.conv.weight[:,:,0,:,:].sum(2).sum(2) + self.conv.weight[:,:,1,:,:].sum(2).sum(2) + self.conv.weight[:,:,3,:,:].sum(2).sum(2) + self.conv.weight[:,:,4,:,:].sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=local_avg, weight=kernel_diff, bias=self.conv.bias, stride=1, padding=0, groups=self.conv.groups)
                return out_normal - self.theta * out_diff
                
            else:
                return out_normal




class T_1DCDC(nn.Module):
    '''
    Temporal Center-difference based Convolutional layer (3D version) 
    theta: control the percentage of original convolution and centeral-difference convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(T_1DCDC, self).__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size,1,1), stride=stride, padding=(padding,1,1), dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            
            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2]>1:
                kernel_diff = self.conv.weight[:,:,0,:,:].sum(2).sum(2) + self.conv.weight[:,:,2,:,:].sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
                return out_normal - self.theta * out_diff
                
            else:
                return out_normal




class T_1DCDC_Avg(nn.Module):
    '''
    Temporal Center-difference based Convolutional layer (3D version) 
    theta: control the percentage of original convolution and centeral-difference convolution
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(T_1DCDC_Avg, self).__init__() 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size,1,1), stride=stride, padding=(padding,1,1), dilation=dilation, groups=groups, bias=bias)
        self.avgpool = nn.AvgPool3d(kernel_size=(kernel_size,1,1), stride=stride, padding=(padding,0,0))
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        local_avg = self.avgpool(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            
            # only CD works on temporal kernel size>1
            if self.conv.weight.shape[2]>1:
                kernel_diff = self.conv.weight[:,:,0,:,:].sum(2).sum(2) + self.conv.weight[:,:,2,:,:].sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None, None]
                out_diff = F.conv3d(input=local_avg, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
                return out_normal - self.theta * out_diff
                
            else:
                return out_normal

class CDC_Unit(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, basic_conv, theta, affine=True):
    super(CDC_Unit, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      basic_conv(C_in, C_out, kernel_size, stride=stride, padding=padding, theta=theta, bias=False),
      nn.BatchNorm3d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

 



class VaniConv3d(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(VaniConv3d, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm3d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)
    
    
class VaniConv3d_Spatial_1x3x3(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(VaniConv3d_Spatial_1x3x3, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv3d(C_in, C_out, [1,3,3], stride=stride, padding=[0,1,1], bias=False),
      nn.BatchNorm3d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)
    
    
class VaniConv3d_Temporal_3x1x1(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(VaniConv3d_Temporal_3x1x1, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv3d(C_in, C_out, [3,1,1], stride=stride, padding=[1,0,0], bias=False),
      nn.BatchNorm3d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)
    
class VaniConv3d_Temporal_5x1x1(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(VaniConv3d_Temporal_5x1x1, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv3d(C_in, C_out, [5,1,1], stride=stride, padding=[2,0,0], bias=False),
      nn.BatchNorm3d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)
    

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm3d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)
    
    
class Dil_STCDC(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, theta, affine=True):
    super(Dil_STCDC, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      ST_3DCDC(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, theta=theta, bias=False),
      nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm3d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)
    
    
class Dil_TCDC(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, theta, affine=True):
    super(Dil_TCDC, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      T_3DCDC(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, theta=theta, bias=False),
      nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm3d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv3d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm3d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm3d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)
    
class Zero_Connection(nn.Module):

  def __init__(self, C_in, C_out, stride):
    super(Zero_Connection, self).__init__()
    self.stride = stride
    self.conv = nn.Conv3d(C_in, C_out, 1, stride=1, padding=0, bias=False) 

  def forward(self, x):
    x = self.conv(x)
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride[0],:,:].mul(0.)


class FactorizedReduce_Spatial(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce_Spatial, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=[1,2,2], padding=0, bias=False)
    self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=[1,2,2], padding=0, bias=False) 
    self.bn = nn.BatchNorm3d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    #out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:, :, 1:,1:])], dim=1)
    out = self.bn(out)
    return out

class FactorizedReduce_SpatialTemporal(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce_SpatialTemporal, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv3d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm3d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    #out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:,1:])], dim=1)
    out = self.bn(out)
    return out
    
    
class MaxPool_ST(nn.Module):

  def __init__(self, stride):
    super(MaxPool_ST, self).__init__()
    self.Maxpool = nn.MaxPool3d(stride, stride)

  def forward(self, x):
    x = self.Maxpool(x)

    return x



class MaxPool_3x1x1(nn.Module):

  def __init__(self, C_in, C_out, stride, affine=True):
    super(MaxPool_3x1x1, self).__init__()
    self.Maxpool = nn.MaxPool3d(kernel_size=[3, 1, 1], padding = [1,0,0], stride=stride)
    self.conv = nn.Conv3d(C_in, C_out, 1, stride=1, padding=0, bias=False) 
    self.bn = nn.BatchNorm3d(C_out, affine=affine)

  def forward(self, x):
    x = self.Maxpool(x)
    x = self.bn(self.conv(x))

    return x
    
class MaxPool_5x1x1(nn.Module):

  def __init__(self, C_in, C_out, stride, affine=True):
    super(MaxPool_5x1x1, self).__init__()
    self.Maxpool = nn.MaxPool3d(kernel_size=[5, 1, 1], padding = [2,0,0], stride=stride)
    self.conv = nn.Conv3d(C_in, C_out, 1, stride=1, padding=0, bias=False) 
    self.bn = nn.BatchNorm3d(C_out, affine=affine)

  def forward(self, x):
    x = self.Maxpool(x)
    x = self.bn(self.conv(x))

    return x
    
    
class AvgPool_3x1x1(nn.Module):

  def __init__(self, C_in, C_out, stride, affine=True):
    super(AvgPool_3x1x1, self).__init__()
    self.Avgpool = nn.AvgPool3d(kernel_size=[3, 1, 1], padding = [1,0,0], stride=stride)
    self.conv = nn.Conv3d(C_in, C_out, 1, stride=1, padding=0, bias=False) 
    self.bn = nn.BatchNorm3d(C_out, affine=affine)

  def forward(self, x):
    x = self.Avgpool(x)
    x = self.bn(self.conv(x))

    return x
    
class AvgPool_5x1x1(nn.Module):

  def __init__(self, C_in, C_out, stride, affine=True):
    super(AvgPool_5x1x1, self).__init__()
    self.Avgpool = nn.AvgPool3d(kernel_size=[5, 1, 1], padding = [2,0,0], stride=stride)
    self.conv = nn.Conv3d(C_in, C_out, 1, stride=1, padding=0, bias=False) 
    self.bn = nn.BatchNorm3d(C_out, affine=affine)

  def forward(self, x):
    x = self.Avgpool(x)
    x = self.bn(self.conv(x))

    return x
