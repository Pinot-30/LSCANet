import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import argparse
import os
#from LDM_Fusion_combine import LDM_Fusion
from LDM_Fusion_combine_res import LDM_Fusion 
#from LDM_Fusion_combine_n import LDM_Fusion
#from LDM_Fusion_combine_res_n import LDM_Fusion
#from LDM_Fusion_combine_nores_n import LDM_Fusion
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from dataset import Dataset
from Eval import *
from collections import defaultdict
import logging

data_transform = {
    'train': torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ]
    ),
    'test': torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5))
        ]
    )
}

device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
test_dataset = Dataset('TNO/', transform=data_transform['test'])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
print(len(test_dataset))
net = LDM_Fusion()

net.load_state_dict(torch.load('Model_res/50.pth',map_location = device))

if not os.path.exists('output_res/'):
    os.makedirs('output_res/')
    pass

net.to(device)

class Histogram(object):
    def __init__(self):
        pass
    # 显示直方图
    def showByHistogram(hist,*,title = ''):
        x_axis = range(0,256)
        y_axis = list(map(lambda x : hist[x] ,x_axis))

        fig = plt.figure(num = title)
        plot_1 = fig.add_subplot(111)
        plot_1.set_title(title)
        plot_1.plot(x_axis,y_axis)
        plt.show()
        del fig

    def show(img,*,title = ''):
        # 只取单通道
        img = img[:,:,0]
        size_h,size_w= img.shape
        logging.info(f' size_h :{size_h} size_w:{size_w} MN:{size_w*size_h}')

        hist = defaultdict(lambda: 0)

        for i in range(size_h):
            for j in range(size_w):
                hist[img[i,j]] += 1
        x_axis = range(0,256)
        y_axis = list(map(lambda x : hist[x] ,x_axis))

        fig = plt.figure(num = title)
        plot_1 = fig.add_subplot(111)
        plot_1.set_title(title)
        plot_1.plot(x_axis,y_axis)
        plt.show()
        del fig
    # 获取直方图，参数Normalized 决定是否归一化
    def get_histogram(img ,*,Normalized = False):
        # 只取 0 通道，若本身只有一个通道不会有影响
        img = img[:,:,0]
        size_h,size_w = img.shape
        logging.info(f' size_h :{size_h} size_w:{size_w} MN:{size_w*size_h}')

        hist = defaultdict(lambda: 0)
        
        for i in range(size_h):
            for j in range(size_w):
                hist[img[i,j]] += 1
                
        # 根据 Normalized 参数决定是否进行归一化
        if Normalized == True:
            sum = 0
            MN = size_h * size_w
            for pixel_value in hist: # Key 迭代
                hist[pixel_value] = hist[pixel_value]/MN
                sum += hist[pixel_value]
            logging.info(f'归一化后加和为：{sum}')
            del sum
        return hist
    # 直方图均衡化函数
    def equalization(img):
        size_h,size_w, size_c = img.shape
        hist = Histogram.get_histogram(img)
        MN = size_h * size_w
        new_hist = defaultdict(lambda: 0)
        # 公式 3.3-8 计算 S_k
        for i in range(0,256):
            for pixel in range(i+1):
                new_hist[i] += hist[pixel]
            new_hist[i] = new_hist[i] * 255/MN
        
        for key in new_hist:
            new_hist[key] = round(new_hist[key])
        new_img = img.copy()
        
        for i in range(new_img.shape[0]):
            for j in range(new_img.shape[1]):
                new_img[i,j] = new_hist[new_img[i,j,0]]

        return new_img
    pass

def padding_4(x, stride = 16):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)
    return out 

def crop_padding_4(fusion, vi_source, stride = 16):
    h, w = vi_source.shape[-2:]
    hf, wf = fusion.shape[2], fusion.shape[3]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    
    out = fusion[:,:,lh:hf-uh,lw:wf-uw]

    return out 

for j in range(len(test_dataset)):
    # Training
    print(j)
    net.eval()
    
    image_ir_source, image_vi_source = test_dataset[j]
    
    image_ir = padding_4(image_ir_source)
    image_vi = padding_4(image_vi_source)
    
#     image_vi = torch.nn.functional.interpolate(image_vi.unsqueeze(0),  scale_factor=0.5, mode='nearest')
#     image_ir = torch.nn.functional.interpolate(image_ir.unsqueeze(0),  scale_factor=0.5, mode='nearest')

    image_vi = image_vi.unsqueeze(0)
    image_ir = image_ir.unsqueeze(0)
    with torch.no_grad(): 
        #image_fusion, cross_att_ir, cross_att_vi, vi_res_att, ir_res_att, vi_encoder_n, ir_encoder_n = net(image_vi.to(device),             image_ir.to(device))
         #image_fusion, cross_att_ir, cross_att_vi, vi_encoder_n, ir_encoder_n = net(image_vi.to(device), image_ir.to(device))
        image_fusion, cross_att_ir, cross_att_vi, vi_res_att, ir_res_att = net(image_vi.to(device), image_ir.to(device))
        
    image_fusion = crop_padding_4(image_fusion, image_vi_source)
          
    
    print(image_fusion.min())
    print(image_fusion.max())
    
    vi_res_att = -1 + 2 * (vi_res_att-vi_res_att.min()) / (vi_res_att.max()-vi_res_att.min())
    ir_res_att = -1 + 2 * (ir_res_att-ir_res_att.min()) / (ir_res_att.max()-ir_res_att.min())    
    
    image_vi_test = TensorToIMage(vi_res_att.cpu().squeeze(0))
    image_ir_test = TensorToIMage(ir_res_att.cpu().squeeze(0))
    imageio.imwrite('output_res_n/vi_res_att/{}_vi_att.bmp'.format(j),image_vi_test)
    imageio.imwrite('output_res_n/ir_res_att/{}_ir_att.bmp'.format(j),image_ir_test)
    
    image_fusion_np = TensorToIMage(image_fusion.cpu().squeeze(0))
    
    if cross_att_ir.shape[1] <= 3:
        
        fea_res_single_np = TensorToIMage(fea_res.cpu().squeeze(0)) 
        imageio.imwrite('output_res_n/FEA_RES_pixil/{}.bmp'.format(j),fea_res_single_np)    
    else:
        for jj in range(16):
            fea_res_single_np_ir = TensorToIMage(cross_att_ir[:,jj:jj+1,:,:].cpu().squeeze(0)) 
            imageio.imwrite('output_res_n/FEA_IR/%d_%d.bmp'%(j, jj),fea_res_single_np_ir)
            fea_res_single_np_vi = TensorToIMage(cross_att_vi[:,jj:jj+1,:,:].cpu().squeeze(0)) 
            imageio.imwrite('output_res_n/FEA_VI/%d_%d.bmp'%(j, jj),fea_res_single_np_vi)
            
            fea_res_single_np_ir = TensorToIMage(vi_encoder_n[:,jj:jj+1,:,:].cpu().squeeze(0)) 
            imageio.imwrite('output_res_n/EN_VI_N/%d_%d.bmp'%(j, jj),fea_res_single_np_ir)
            fea_res_single_np_vi = TensorToIMage(ir_encoder_n[:,jj:jj+1,:,:].cpu().squeeze(0)) 
            imageio.imwrite('output_res_n/EN_IR_N/%d_%d.bmp'%(j, jj),fea_res_single_np_vi)
            
#         for jjj in range(3):
#             fea_res_rgb = torch.cat((fea_res[:,jjj:jjj+1,:,:], fea_res[:,jjj+10:jjj+1+10,:,:], fea_res[:,jjj+20:jjj+1+20,:,:]), 1)
#             fea_res_rgb_np = TensorToIMage(fea_res_rgb.cpu().squeeze(0)) 
#             imageio.imwrite('output/FEA_RES_RGB/%d_%d.bmp'%(j, jjj),fea_res_rgb_np)
#         fea_res_rgb = torch.cat((torch.mean(fea_res[:, 0:10, :, :], 1, keepdim=True), torch.mean(fea_res[:, 10:20, :, :], 1, keepdim=True), torch.mean(fea_res[:, 20:30, :, :], 1, keepdim=True)), 1)
#         fea_res_rgb_np = TensorToIMage(fea_res_rgb.cpu().squeeze(0)) 
#         imageio.imwrite('output/FEA_RES_RGB/%d.bmp'%(j),fea_res_rgb_np) 
#     image_fusion_np = Histogram.equalization(torch.tensor(image_fusion_np).unsqueeze(2).numpy())
#     imageio.imwrite('output/fusion/{}.bmp'.format(j),torch.tensor(image_fusion_np).squeeze().numpy())
   
    imageio.imwrite('output_res_n/fusion/{}.bmp'.format(j),image_fusion_np)    
    
    #imageio.imwrite('output/{}_fea.bmp'.format(j),fea_1)
    pass

print('Finished')