import os
from evaluate import evaluate
from config import common_config
import torch
from nets.cnn_ctc import CNN
from dataset import get_loader
from torch.nn import CTCLoss
import time

device = common_config['device']
base_path = './checkpoint'
ckpt = 'cnn_156000_loss0.162173.pth'
# ckpts = os.listdir(base_path)
train_loader, val_loader = get_loader()
criterion = CTCLoss(blank=0, reduction='sum')
# print(ckpts)
# ('crnn_11000_loss0.055960.pth', 0.9878595143805752)
# for ckpt in ckpts:
#     pretrained_path = os.path.join(base_path, ckpt)
#     net = CRNN(input_c=common_config['img_channel'],
#                input_h=common_config['img_height'], num_classes=common_config['num_classes'] + 1)
#     net.load_state_dict(torch.load(pretrained_path, map_location=device))
#     net.to(device)
#     metrics = evaluate(net, val_loader, criterion, device)
#     print((ckpt, metrics['acc']))
pretrained_path = os.path.join(base_path,ckpt)
print(ckpt)
net = CNN(input_c=common_config['img_channel'],
input_h=common_config['img_height'], num_classes=common_config['num_classes'] + 1)    #这里比较奇怪   没有用softmax
net.load_state_dict(torch.load(pretrained_path, map_location=device))
net.to(device)
clock_start=time.time()
metrics = evaluate(net, val_loader, criterion, device)
print(ckpt, metrics['acc'])
print('整个测试耗时{}S'.format(time.time()-clock_start))
