from os import path
import torch
import glob
from torch.nn.functional import log_softmax
from torchvision import transforms
from nets.cnn_ctc import CNN
from config import common_config
from dataset import LicensePlate
from ctc_decoder import ctc_decode
from torch.utils.data import DataLoader
import time
from PIL import Image
import numpy as np
from pytorch_model_summary import summary
import ctypes as cts


pretrained_path = '/home/rzhang/Documents/project/ocr/checkpoint/ckpoint3/cnn_73500_loss0.196247.pth'     ####'./checkpoint/cnn_63000_loss0.196440.pth'

decode_method = 'greedy'#'beam_search'
beam_size = 10
device = common_config['device']
transform = transforms.Compose(
    [transforms.Resize([32, 100]),
     transforms.ToTensor(),  # 将PIL或者ndarray转换为0~1之间的tensor
     transforms.Normalize((0.5), (0.5))]  # 对每个通道将像素值缩放到-1~1之间
)

def batch_prediction(path):
    print('-----------批量图片识别开始-----------')
    print(device)
    images_path = glob.glob(path)
    batch_size = 32

    demo_dataset = LicensePlate(paths=images_path, transform=transform)
    demo_loader = DataLoader(dataset=demo_dataset,
                             batch_size=batch_size, shuffle=False)
    net = CNN(input_c=common_config['img_channel'],
               input_h=common_config['img_height'], num_classes=common_config['num_classes'] + 1)
    net.load_state_dict(torch.load(pretrained_path, map_location=device))
    net.to(device)
    # outputs = []
    # clock_start = time.time()
    with torch.no_grad():
        for data in demo_loader:
            images = data.to(device)
            so=cts.cdll.LoadLibrary("./decode.so")
            logits = net(images)
            logits=np.transpose(logits.cpu().numpy(), (1, 0, 2))
            for l in logits:
                # row,col=l.shape    #14 67
                l=l.astype(np.double)
                p=l.ctypes.data_as(cts.POINTER(cts.c_double))
                so.decode(p)
       
    
def single_prediction(path):
    print('-----------单张图片识别开始-----------')
    print(device)
    # image = Image.open(path).convert('L')
    image = Image.open(path).convert('RGB')   #注意这里改成了RGB输入
    image = transform(image).unsqueeze(0)  # 增加batch维度
 
    net = CNN(input_c=common_config['img_channel'],
               input_h=common_config['img_height'], num_classes=common_config['num_classes'] + 1)
    net.load_state_dict(torch.load(pretrained_path, map_location=device))
    net.to(device)
   
    with torch.no_grad():
        image = image.to(device)
        # clock_start = time.time()
        logits = net(image)
 
        
        logits=np.transpose(logits.cpu().numpy(), (1, 0, 2))
        for l in logits:
            # row,col=l.shape    #14 67
            l=l.astype(np.double)
            so=cts.cdll.LoadLibrary("./decode.so")
            # so.decode.restype = cts.POINTER(struct_pointer)
            p=l.ctypes.data_as(cts.POINTER(cts.c_double))
            so.decode(p)
      

def predict(path, mode='single'):
    '''
    single模式下,path即为图片路径
    batch模式下,path为 './demo/*.jpg'格式
    '''
    if mode == 'single':
        single_prediction(path)
    elif mode == 'batch':
        batch_prediction(path)

if __name__ == '__main__':
    
    
    
    path = '/home/rzhang/Documents/project/ocr/31_皖B76840.jpg'
    predict(path, mode='single')
    
    # path = './demo/*.jpg'
    # predict(path, mode='batch')

