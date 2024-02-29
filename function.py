import numpy as np
import skimage
from skimage import io, measure
import random
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preclassify import del2, srad, dicomp, FCM, hcluster
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import base64
from collections import  Counter

def image_padding(data,r):
    if len(data.shape)==3:
        data_new=np.lib.pad(data,((r,r),(r,r),(0,0)),'constant',constant_values=0)
        return data_new
    if len(data.shape)==2:
        data_new=np.lib.pad(data,r,'constant',constant_values=0)
        return data_new

def createTestingCubes(X, patch_size):
    # 给 X 做 padding
    margin = int((patch_size - 1) / 2)
    zeroPaddedX = image_padding(X, margin)
    patchesData = np.zeros( (X.shape[0]*X.shape[1], patch_size, patch_size, X.shape[2]) )
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchIndex = patchIndex + 1
    return patchesData   

def postprocess(res):
    res_new = res
    res = measure.label(res, connectivity=2)
    num = res.max()
    for i in range(1, num+1):
        idy, idx = np.where(res==i)
        if len(idy) <= 20:
            res_new[idy, idx] = 0
    return res_new

class MRC(nn.Module):
    def __init__(self, inchannel):
        super(MRC, self).__init__() 
        self.conv1 = nn.Conv2d(inchannel, 15, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(15)

        self.conv2_1 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(5)

        self.conv2_2 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(5)

        self.conv2_3 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_3 = nn.BatchNorm2d(5)

    def forward(self, x):
        ori_out = F.relu(self.bn1(self.conv1(x)))

        shape=(x.shape[0], 5, 7, 7)

        all_zero3_3=torch.zeros(size=shape).cuda()
        all_zero1_3=torch.zeros(size=(x.shape[0], 5, 3, 7)).cuda()
        all_zero3_1=torch.zeros(size=(x.shape[0], 5, 7, 3)).cuda()

        all_zero3_3[:,:,:,:]=ori_out[:,0:5,:,:]
        all_zero1_3[:,:,:,:]=ori_out[:,5:10,2:5,:]
        all_zero3_1[:,:,:,:]=ori_out[:,10:15,:,2:5]

        square=F.relu(self.bn2_1(self.conv2_1(all_zero3_3)))
        horizontal=F.relu(self.bn2_2(self.conv2_2(all_zero1_3)))
        vertical=F.relu(self.bn2_3(self.conv2_3(all_zero3_1)))
        horizontal_final=torch.zeros(size=(x.shape[0], 5, 7, 7)).cuda()
        vertical_final=torch.zeros(size=(x.shape[0], 5, 7, 7)).cuda()
        horizontal_final[:,:,2:5,:]=horizontal[:,:,:,:]
        vertical_final[:,:,:,2:5]=vertical[:,:,:,:]

        glo = square + horizontal_final + vertical_final
        #glo= F.relu(self.bn3(self.conv3(glo)))
        
        return glo

def DCT(x):
      out=F.interpolate(x, size=(8,8), mode='bilinear', align_corners=True)
      #print(out.shape)
      #dct_out_1 =torch.Tensor([cv2.dct(x[i,0,:,:].detach().cpu().numpy()) \
      #                          for i in range(x.shape[0])])
      dct_out_1 =torch.Tensor([cv2.dct(np.float32(out[i,0,:,:].detach().cpu().numpy())) \
                                for i in range(x.shape[0])])
      dct_out_2 =torch.Tensor([cv2.dct(np.float32(out[i,1,:,:].detach().cpu().numpy())) \
                                for i in range(x.shape[0])])
      dct_out_3 =torch.Tensor([cv2.dct(np.float32(out[i,2,:,:].detach().cpu().numpy())) \
                                for i in range(x.shape[0])])
      dct_out=torch.zeros(size=(x.shape[0],3, 8, 8))
      dct_out[:,0,:,:]=dct_out_1 
      dct_out[:,1,:,:]=dct_out_2
      dct_out[:,2,:,:]=dct_out_3
      dct_out=dct_out.cuda()#放回cuda
      out=dct_out.view(x.shape[0], 3, 64)
      #out=torch.cat((out,out),2)
      out=F.glu(out,dim=-1)
      dct_out=out.view(x.shape[0], 1, 96)
      return dct_out
############################################################################################################################################
class DDNet(nn.Module):
    def __init__(self):
        super(DDNet, self).__init__() 
        self.mrc1=MRC(3)
        self.mrc2=MRC(5)
        self.mrc3=MRC(5)
        self.mrc4=MRC(5)

        self.linear1=nn.Linear(341, 10) 
        self.linear2=nn.Linear(10, 2)

    def forward(self, x):

        m_1=self.mrc1(x)
        m_2=self.mrc2(m_1)
        m_3=self.mrc3(m_2)
        m_4=self.mrc4(m_3)
        #glo= F.relu(self.bn(self.conv(m_4)))
        glo=m_4.view(x.shape[0], 1, 245)

        dct_out=DCT(x)
        
        out=torch.cat((glo,dct_out),2)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out_1 = self.linear1(out)
        out = self.linear2(out_1)

        return out    