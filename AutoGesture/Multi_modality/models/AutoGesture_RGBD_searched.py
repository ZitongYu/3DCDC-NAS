import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
import pdb


class Cell_Con(nn.Module):

  def __init__(self, op_names, C_in, C_out, stride):
    super(Cell_Con, self).__init__()
    
    self.op = OPS_connection[op_names](C_in, C_out, stride, True)

  def forward(self, inputs):
    outputs = self.op(inputs)
    
    return outputs






class Cell_Net(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction_prev, normal):
    super(Cell_Net, self).__init__()
    
    # stride = [1,2,2]
    if reduction_prev:
      self.preprocess0 = VaniConv3d(C_prev_prev, C, 1, [1,2,2], 0, affine=False)  
    else:
      self.preprocess0 = VaniConv3d(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = VaniConv3d(C_prev, C, 1, 1, 0, affine=False)
    
    # branch 8 frame
    if normal==1:
        op_names, indices = zip(*genotype.normal8)
        concat = genotype.normal_concat8
    # branch 16 frame
    if normal==2:
        op_names, indices = zip(*genotype.normal16)
        concat = genotype.normal_concat16
    # branch 32 frame
    if normal==3:
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
    
    #pdb.set_trace()
    
    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)
    




# steps = how many nodes in a cell; multiplier = how many nodes to be concated 
# Default, layer = 12, steps = 4, multiplier = 4
# [3 x 32 x 112 x 112]
# 12 layers,  layers [0,3,8] with one MaxPool(stride=[1,2,2])
class AutoGesture_RGBD_12layers(nn.Module):
  def __init__(self, C, num_classes, layers, genotype_searched_RGB, genotype_searched_Depth, genotype_con_unshared):  
    super(AutoGesture_RGBD_12layers, self).__init__()
    
    self._C = C    # 8
    self._num_classes = num_classes
    self._layers = layers
    
    self._genotype_searched_RGB = genotype_searched_RGB
    self._genotype_searched_Depth = genotype_searched_Depth
    self._genotype_con_unshared = genotype_con_unshared

    
    self.MaxpoolSpa = nn.MaxPool3d(kernel_size=[1, 3, 3], padding = [0,1,1], stride=[1, 2, 2])
    self.AvgpoolSpa = nn.AvgPool3d(kernel_size=2, stride=2)
    
    
    #  (0) R16-> R8   (1) R32-> R16   (2)  R32-> R8 (s=4)
    #  (3) D16-> D8   (4) D32-> D16   (5)  D32-> D8 (s=4)
    #  (6) R8-> D8  (s=1)  (7) R16-> D8    (8) R16-> D16 (s=1)  (9)  R32-> D8 (s=4) (10)  R32-> D16  (11)  R32-> D32  (s=1)
    #  (12)D8-> R8  (s=1) (13) D16-> R8   (14) D16-> R16 (s=1) (15)  D32-> R8 (s=4)(16)  D32-> R16  (17)  D32-> R32   (s=1)
    
    
    self.cells_Low_Connect = nn.ModuleList()
    self.cells_Mid_Connect = nn.ModuleList()
    self.cells_High_Connect = nn.ModuleList()
    for i in range(18):    # 18 connections
      if i in [6,8,11,12,14,17]:    # stride = 1
        cell_low = Cell_Con(self._genotype_con_unshared.Low_Connect[i], C*2, C, [1,1,1])
        cell_mid = Cell_Con(self._genotype_con_unshared.Mid_Connect[i], C*2*4, C*2, [1,1,1])
        cell_high = Cell_Con(self._genotype_con_unshared.High_Connect[i], C*4*4, C*4, [1,1,1])
        
      elif i in [2,5,9,15]:    # stride = 4
        cell_low = Cell_Con(self._genotype_con_unshared.Low_Connect[i], C*2, C, [4,1,1])
        cell_mid = Cell_Con(self._genotype_con_unshared.Mid_Connect[i], C*2*4, C*2, [4,1,1])
        cell_high = Cell_Con(self._genotype_con_unshared.High_Connect[i], C*4*4, C*4, [4,1,1])
        
      else:      # stride = 2
        cell_low = Cell_Con(self._genotype_con_unshared.Low_Connect[i], C*2, C, [2,1,1])
        cell_mid = Cell_Con(self._genotype_con_unshared.Mid_Connect[i], C*2*4, C*2, [2,1,1])
        cell_high = Cell_Con(self._genotype_con_unshared.High_Connect[i], C*4*4, C*4, [2,1,1])

      self.cells_Low_Connect += [cell_low]
      self.cells_Mid_Connect += [cell_mid]
      self.cells_High_Connect += [cell_high]
        
    
    
    # 8 frames
    self.stem0_8_RGB = nn.Sequential(
      nn.Conv3d(3, C, kernel_size=3, stride=1, padding=1, bias=False),  
      nn.BatchNorm3d(C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=[1, 3, 3], padding = [0,1,1], stride=[1, 2, 2]),
    )
    self.stem1_8_RGB = nn.Sequential(
      
      nn.Conv3d(C, 2*C, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm3d(2*C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=3, padding = 1, stride=2),
    )
    self.stem0_8_Depth = nn.Sequential(
      nn.Conv3d(3, C, kernel_size=3, stride=1, padding=1, bias=False),  
      nn.BatchNorm3d(C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=[1, 3, 3], padding = [0,1,1], stride=[1, 2, 2]),
    )
    self.stem1_8_Depth = nn.Sequential(
      
      nn.Conv3d(C, 2*C, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm3d(2*C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=3, padding = 1, stride=2),
    )
    
    
    # 16 frames
    self.stem0_16_RGB = nn.Sequential(
      nn.Conv3d(3, C, kernel_size=3, stride=1, padding=1, bias=False),  
      nn.BatchNorm3d(C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=[1, 3, 3], padding = [0,1,1], stride=[1, 2, 2]),
    )
    self.stem1_16_RGB = nn.Sequential(
      
      nn.Conv3d(C, 2*C, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm3d(2*C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=3, padding = 1, stride=2),
    )
    self.stem0_16_Depth = nn.Sequential(
      nn.Conv3d(3, C, kernel_size=3, stride=1, padding=1, bias=False),  
      nn.BatchNorm3d(C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=[1, 3, 3], padding = [0,1,1], stride=[1, 2, 2]),
    )
    self.stem1_16_Depth = nn.Sequential(
      
      nn.Conv3d(C, 2*C, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm3d(2*C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=3, padding = 1, stride=2),
    )
    
    # 32 frames
    self.stem0_32_RGB = nn.Sequential(
      nn.Conv3d(3, C, kernel_size=3, stride=1, padding=1, bias=False),  
      nn.BatchNorm3d(C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=[1, 3, 3], padding = [0,1,1], stride=[1, 2, 2]),
    )
    self.stem1_32_RGB = nn.Sequential(
      
      nn.Conv3d(C, 2*C, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm3d(2*C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=3, padding = 1, stride=2),
    )
    self.stem0_32_Depth = nn.Sequential(
      nn.Conv3d(3, C, kernel_size=3, stride=1, padding=1, bias=False),  
      nn.BatchNorm3d(C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=[1, 3, 3], padding = [0,1,1], stride=[1, 2, 2]),
    )
    self.stem1_32_Depth = nn.Sequential(
      
      nn.Conv3d(C, 2*C, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm3d(2*C),
      nn.ReLU(inplace=True),
      nn.MaxPool3d(kernel_size=3, padding = 1, stride=2),
    )
    
     
 
    C_prev_prev8, C_prev8, C_curr8 = C, 7*C, 2*C
    C_prev_prev16, C_prev16, C_curr16 = C, 5*C, 2*C
    C_prev_prev32, C_prev32, C_curr32 = C, 3*C, 2*C
    
    
    
    ##  8  ##
    self.cells8_RGB = nn.ModuleList()
    self.cells8_Depth = nn.ModuleList()
    normal = 1
    reduction_prev = False
    for i in range(layers):
      if i in [4,9]:
        C_curr8 *= 2
        reduction_prev = True
        
        if i==4:
            C_prev8 = 4*C*4 + 2*C  #144
        
        if i==9:
            C_prev8 = 8*C*4 + 4*C
        
      else:
        reduction_prev = False
      cell_RGB = Cell_Net(genotype_searched_RGB, C_prev_prev8, C_prev8, C_curr8, reduction_prev, normal)
      cell_Depth = Cell_Net(genotype_searched_Depth, C_prev_prev8, C_prev8, C_curr8, reduction_prev, normal)
      
      self.cells8_RGB += [cell_RGB]
      self.cells8_Depth += [cell_Depth]
      
      C_prev_prev8, C_prev8 = C_prev8, cell_RGB.multiplier*C_curr8
    
    
    ##  16  ##
    self.cells16_RGB = nn.ModuleList()
    self.cells16_Depth = nn.ModuleList()
    normal = 2
    reduction_prev = False
    for i in range(layers):
      if i in [4,9]:
        C_curr16 *= 2
        reduction_prev = True
        
        if i==4:
            C_prev16 = 3*C*4 + 2*C  #112
        
        if i==9:
            C_prev16 = 6*C*4 + 4*C
      else:
        reduction_prev = False
      cell_RGB = Cell_Net(genotype_searched_RGB, C_prev_prev16, C_prev16, C_curr16, reduction_prev, normal)
      cell_Depth = Cell_Net(genotype_searched_Depth, C_prev_prev16, C_prev16, C_curr16, reduction_prev, normal)
      
      self.cells16_RGB += [cell_RGB]
      self.cells16_Depth += [cell_Depth]
      
      C_prev_prev16, C_prev16 = C_prev16, cell_RGB.multiplier*C_curr16
      
      
    ##  32  ##
    self.cells32_RGB = nn.ModuleList()
    self.cells32_Depth = nn.ModuleList()
    normal = 3
    reduction_prev = False
    for i in range(layers):
      if i in [4,9]:
        C_curr32 *= 2
        reduction_prev = True
        
        if i==4:
            C_prev32 = 2*C*4 + 2*C  #80
        
        if i==9:
            C_prev32 = 4*C*4 + 4*C
      else:
        reduction_prev = False
      
      cell_RGB = Cell_Net(genotype_searched_RGB, C_prev_prev32, C_prev32, C_curr32, reduction_prev, normal)
      cell_Depth = Cell_Net(genotype_searched_Depth, C_prev_prev32, C_prev32, C_curr32, reduction_prev, normal)
      
      self.cells32_RGB += [cell_RGB]
      self.cells32_Depth += [cell_Depth]
      
      C_prev_prev32, C_prev32 = C_prev32, cell_RGB.multiplier*C_curr32
    
    
    # head
    self.global_pooling = nn.AdaptiveAvgPool3d(1)
    self.dropout = nn.Dropout(p=0.5)
    self.classifier = nn.Linear(C_prev8*2+C_prev16*2+C_prev32*2, num_classes)
    
    
    

  def forward(self, inputs, inputs_D):
  
    ## inputs 32 x 3 x 112 x 112 ##
    
    
    # 32
    s0_32_RGB = self.stem0_32_RGB(inputs)   # 32*28*28 
    s1_32_RGB = self.stem1_32_RGB(s0_32_RGB)       # 16*28*28  
    s0_32_RGB = self.AvgpoolSpa(s0_32_RGB)  # 16*28*28 
    
    s0_32_Depth = self.stem0_32_Depth(inputs_D)   # 32*28*28 
    s1_32_Depth = self.stem1_32_Depth(s0_32_Depth)       # 16*28*28  
    s0_32_Depth = self.AvgpoolSpa(s0_32_Depth)  # 16*28*28 
    
    
    
    # 16
    s0_16_RGB = self.stem0_16_RGB(inputs[:, :, ::2, :, :])   # 16*28*28 
    s1_16_RGB = self.stem1_16_RGB(s0_16_RGB)       # 8*28*28  
    s0_16_RGB = self.AvgpoolSpa(s0_16_RGB)  # 8*28*28 
    
    s0_16_Depth = self.stem0_16_Depth(inputs_D[:, :, ::2, :, :])   # 16*28*28 
    s1_16_Depth = self.stem1_16_Depth(s0_16_Depth)       # 8*28*28  
    s0_16_Depth = self.AvgpoolSpa(s0_16_Depth)  # 8*28*28 
    
    
    
    # 8 
    s0_8_RGB = self.stem0_8_RGB(inputs[:, :, ::4, :, :])   # 8*28*28 
    s1_8_RGB = self.stem1_8_RGB(s0_8_RGB)       # 4*28*28  
    s0_8_RGB = self.AvgpoolSpa(s0_8_RGB)  # 4*28*28 
    
    s0_8_Depth = self.stem0_8_Depth(inputs_D[:, :, ::4, :, :])   # 8*28*28 
    s1_8_Depth = self.stem1_8_Depth(s0_8_Depth)       # 4*28*28  
    s0_8_Depth = self.AvgpoolSpa(s0_8_Depth)  # 4*28*28 
    
    
    #  (0) R16-> R8   (1) R32-> R16   (2)  R32-> R8 (s=4)
    #  (3) D16-> D8   (4) D32-> D16   (5)  D32-> D8 (s=4)
    #  (6) R8-> D8  (s=1)  (7) R16-> D8    (8) R16-> D16 (s=1)  (9)  R32-> D8 (s=4) (10)  R32-> D16  (11)  R32-> D32  (s=1)
    #  (12)D8-> R8  (s=1) (13) D16-> R8   (14) D16-> R16 (s=1) (15)  D32-> R8 (s=4)(16)  D32-> R16  (17)  D32-> R32   (s=1)
    
    
    laterals_Low = []
    #for i in range(18):  # 18 connections
    for i, cell in enumerate(self.cells_Low_Connect):
      #pdb.set_trace()
      if i in [1,2,9,10,11]:    # RGB32
        lateral1 = cell(s1_32_RGB)
        
      elif i in [0,7,8]:    # RGB16
        lateral1 = cell(s1_16_RGB)
      
      elif i in [6]:    # RGB8
        lateral1 = cell(s1_8_RGB)
      
      elif i in [4,5,15,16,17]:    # Depth32
        lateral1 = cell(s1_32_Depth)
      
      elif i in [3,13,14]:    # Depth16
        lateral1 = cell(s1_16_Depth)
        
      elif i in [12]:    # Depth8
        lateral1 = cell(s1_8_Depth)
        
      laterals_Low.append(lateral1)
    
    s1_32_RGB = torch.cat([laterals_Low[17], s1_32_RGB],dim=1)
    s1_16_RGB = torch.cat([laterals_Low[1],laterals_Low[14],laterals_Low[16], s1_16_RGB],dim=1)
    s1_8_RGB = torch.cat([laterals_Low[0],laterals_Low[2],laterals_Low[12],laterals_Low[13],laterals_Low[15], s1_8_RGB],dim=1)
    
    s1_32_Depth = torch.cat([laterals_Low[11], s1_32_Depth],dim=1)
    s1_16_Depth = torch.cat([laterals_Low[4],laterals_Low[8],laterals_Low[10], s1_16_Depth],dim=1)
    s1_8_Depth = torch.cat([laterals_Low[3],laterals_Low[5],laterals_Low[6],laterals_Low[7],laterals_Low[9], s1_8_Depth],dim=1)
    
    
    # 8 layers
    for ii in range(0,4):
      s0_32_RGB, s1_32_RGB = s1_32_RGB, self.cells32_RGB[ii](s0_32_RGB, s1_32_RGB)
      s0_32_Depth, s1_32_Depth = s1_32_Depth, self.cells32_Depth[ii](s0_32_Depth, s1_32_Depth)
      
      s0_16_RGB, s1_16_RGB = s1_16_RGB, self.cells16_RGB[ii](s0_16_RGB, s1_16_RGB)
      s0_16_Depth, s1_16_Depth = s1_16_Depth, self.cells16_Depth[ii](s0_16_Depth, s1_16_Depth)
      
      s0_8_RGB, s1_8_RGB = s1_8_RGB, self.cells8_RGB[ii](s0_8_RGB, s1_8_RGB)
      s0_8_Depth, s1_8_Depth = s1_8_Depth, self.cells8_Depth[ii](s0_8_Depth, s1_8_Depth)
    
      if ii==3:
        x_visual28 = s1_16_RGB
        
        s1_32_RGB = self.MaxpoolSpa(s1_32_RGB) 
        s1_32_Depth = self.MaxpoolSpa(s1_32_Depth) 
        s1_16_RGB = self.MaxpoolSpa(s1_16_RGB) 
        s1_16_Depth = self.MaxpoolSpa(s1_16_Depth) 
        s1_8_RGB = self.MaxpoolSpa(s1_8_RGB) 
        s1_8_Depth = self.MaxpoolSpa(s1_8_Depth) 
    
    
    
    laterals_Mid = []
    for i in range(18):  # 18 connections
      
      if i in [1,2,9,10,11]:    # RGB32
        lateral1 = self.cells_Mid_Connect[i](s1_32_RGB)
        
      elif i in [0,7,8]:    # RGB16
        lateral1 = self.cells_Mid_Connect[i](s1_16_RGB)
      
      elif i in [6]:    # RGB8
        lateral1 = self.cells_Mid_Connect[i](s1_8_RGB)
      
      elif i in [4,5,15,16,17]:    # Depth32
        lateral1 = self.cells_Mid_Connect[i](s1_32_Depth)
      
      elif i in [3,13,14]:    # Depth16
        lateral1 = self.cells_Mid_Connect[i](s1_16_Depth)
        
      elif i in [12]:    # Depth8
        lateral1 = self.cells_Mid_Connect[i](s1_8_Depth)
        
      laterals_Mid.append(lateral1)
    
    
    
    
    s1_32_RGB = torch.cat([laterals_Mid[17], s1_32_RGB],dim=1)
    s1_16_RGB = torch.cat([laterals_Mid[1],laterals_Mid[14],laterals_Mid[16], s1_16_RGB],dim=1)
    s1_8_RGB = torch.cat([laterals_Mid[0],laterals_Mid[2],laterals_Mid[12],laterals_Mid[13],laterals_Mid[15], s1_8_RGB],dim=1)
    
    s1_32_Depth = torch.cat([laterals_Mid[11], s1_32_Depth],dim=1)
    s1_16_Depth = torch.cat([laterals_Mid[4],laterals_Mid[8],laterals_Mid[10], s1_16_Depth],dim=1)
    s1_8_Depth = torch.cat([laterals_Mid[3],laterals_Mid[5],laterals_Mid[6],laterals_Mid[7],laterals_Mid[9], s1_8_Depth],dim=1)
    
    
    
    #--------------------------------------------------------------------------------
    
    
    for ii in range(4,9):
      s0_32_RGB, s1_32_RGB = s1_32_RGB, self.cells32_RGB[ii](s0_32_RGB, s1_32_RGB)
      s0_32_Depth, s1_32_Depth = s1_32_Depth, self.cells32_Depth[ii](s0_32_Depth, s1_32_Depth)
      
      s0_16_RGB, s1_16_RGB = s1_16_RGB, self.cells16_RGB[ii](s0_16_RGB, s1_16_RGB)
      s0_16_Depth, s1_16_Depth = s1_16_Depth, self.cells16_Depth[ii](s0_16_Depth, s1_16_Depth)
      
      s0_8_RGB, s1_8_RGB = s1_8_RGB, self.cells8_RGB[ii](s0_8_RGB, s1_8_RGB)
      s0_8_Depth, s1_8_Depth = s1_8_Depth, self.cells8_Depth[ii](s0_8_Depth, s1_8_Depth)
    
      if ii==8:
        x_visual14 = s1_16_RGB
        
        s1_32_RGB = self.MaxpoolSpa(s1_32_RGB) 
        s1_32_Depth = self.MaxpoolSpa(s1_32_Depth) 
        s1_16_RGB = self.MaxpoolSpa(s1_16_RGB) 
        s1_16_Depth = self.MaxpoolSpa(s1_16_Depth) 
        s1_8_RGB = self.MaxpoolSpa(s1_8_RGB) 
        s1_8_Depth = self.MaxpoolSpa(s1_8_Depth) 
    
    
    
    laterals_High = []
    for i in range(18):  # 18 connections

      if i in [1,2,9,10,11]:    # RGB32
        lateral1 = self.cells_High_Connect[i](s1_32_RGB)
        
      elif i in [0,7,8]:    # RGB16
        lateral1 = self.cells_High_Connect[i](s1_16_RGB)
      
      elif i in [6]:    # RGB8
        lateral1 = self.cells_High_Connect[i](s1_8_RGB)
      
      elif i in [4,5,15,16,17]:    # Depth32
        lateral1 = self.cells_High_Connect[i](s1_32_Depth)
      
      elif i in [3,13,14]:    # Depth16
        lateral1 = self.cells_High_Connect[i](s1_16_Depth)
        
      elif i in [12]:    # Depth8
        lateral1 = self.cells_High_Connect[i](s1_8_Depth)
        
      laterals_High.append(lateral1)
      
    
    
    s1_32_RGB = torch.cat([laterals_High[17], s1_32_RGB],dim=1)
    s1_16_RGB = torch.cat([laterals_High[1],laterals_High[14],laterals_High[16], s1_16_RGB],dim=1)
    s1_8_RGB = torch.cat([laterals_High[0],laterals_High[2],laterals_High[12],laterals_High[13],laterals_High[15], s1_8_RGB],dim=1)
    
    s1_32_Depth = torch.cat([laterals_High[11], s1_32_Depth],dim=1)
    s1_16_Depth = torch.cat([laterals_High[4],laterals_High[8],laterals_High[10], s1_16_Depth],dim=1)
    s1_8_Depth = torch.cat([laterals_High[3],laterals_High[5],laterals_High[6],laterals_High[7],laterals_High[9], s1_8_Depth],dim=1)
    
    
    #--------------------------------------------------------------------------------
        
    for ii in range(9,12):
      s0_32_RGB, s1_32_RGB = s1_32_RGB, self.cells32_RGB[ii](s0_32_RGB, s1_32_RGB)
      s0_32_Depth, s1_32_Depth = s1_32_Depth, self.cells32_Depth[ii](s0_32_Depth, s1_32_Depth)
      
      s0_16_RGB, s1_16_RGB = s1_16_RGB, self.cells16_RGB[ii](s0_16_RGB, s1_16_RGB)
      s0_16_Depth, s1_16_Depth = s1_16_Depth, self.cells16_Depth[ii](s0_16_Depth, s1_16_Depth)
      
      s0_8_RGB, s1_8_RGB = s1_8_RGB, self.cells8_RGB[ii](s0_8_RGB, s1_8_RGB)
      s0_8_Depth, s1_8_Depth = s1_8_Depth, self.cells8_Depth[ii](s0_8_Depth, s1_8_Depth)
    

    #--------------------------------------------------------------------------------      
    
    out_32_RGB = self.global_pooling(s1_32_RGB)   # C*8*4*=8*8*4=256       
    out_16_RGB = self.global_pooling(s1_16_RGB)   # 
    out_8_RGB = self.global_pooling(s1_8_RGB)    # 
    
    out_32_Depth = self.global_pooling(s1_32_Depth)   # C*8*4*=8*8*4=256       
    out_16_Depth = self.global_pooling(s1_16_Depth)   # 
    out_8_Depth = self.global_pooling(s1_8_Depth)    # 
    
    out = torch.cat([out_32_RGB, out_16_RGB, out_8_RGB, out_32_Depth, out_16_Depth, out_8_Depth],dim=1)
    logits = self.classifier(self.dropout(out.view(out.size(0),-1)))  # dropout=0.5
    #logits = self.classifier(out.view(out.size(0),-1))
    
    #return logits, x_visual28, x_visual14
    return logits
    
    
    
    
    
    
    
    