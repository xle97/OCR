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
            so=cts.cdll.LoadLibrary("./test.so")
            logits = net(images)
            logits=np.transpose(logits.cpu().numpy(), (1, 0, 2))
            for l in logits:
                # row,col=l.shape    #14 67
                l=l.astype(np.double)
                p=l.ctypes.data_as(cts.POINTER(cts.c_double))
                so.decode(p)
            # log_probs = log_softmax(logits)  # 这一句不要
    #         preds = ctc_decode(logits, method=decode_method,
    #                            beam_size=beam_size, label2char=LicensePlate.LABEL2CHAR)
    #         outputs += preds
    # print('图片推理平均耗时{:.6f}s'.format((time.time() - clock_start) / len(outputs)))
    # i = 0
    # for path, pred in zip(images_path, outputs):
    #     if i > 1000:
    #         break
    #     text = ''.join(pred)
    #     print('{}的预测结果为：{}'.format(path, text))
    #     i += 1

# class struct_pointer(cts.Structure):
#     _fields_=[('ans',cts.c_int*10)]
    
def single_prediction(path):
    print('-----------单张图片识别开始-----------')
    print(device)
    # image = Image.open(path).convert('L')
    image = Image.open(path).convert('RGB')   #注意这里改成了RGB输入
    image = transform(image).unsqueeze(0)  # 增加batch维度
    # print(type(image),image.shape)
    net = CNN(input_c=common_config['img_channel'],
               input_h=common_config['img_height'], num_classes=common_config['num_classes'] + 1)
    net.load_state_dict(torch.load(pretrained_path, map_location=device))
    net.to(device)
   
    with torch.no_grad():
        image = image.to(device)
        # clock_start = time.time()
        logits = net(image)
        # print('单张图片推理耗时{}s'.format(time.time()-clock_start))
        # log_probs = log_softmax(logits, dim=2)
        
        logits=np.transpose(logits.cpu().numpy(), (1, 0, 2))
        for l in logits:
            # row,col=l.shape    #14 67
            l=l.astype(np.double)
            so=cts.cdll.LoadLibrary("./test.so")
            # so.decode.restype = cts.POINTER(struct_pointer)
            p=l.ctypes.data_as(cts.POINTER(cts.c_double))
            so.decode(p)
        # pred = ctc_decode(logits, method=decode_method,
        #                   beam_size=beam_size, label2char=LicensePlate.LABEL2CHAR)
        # text = ''.join(pred[0])
        # print('{}的预测结果为：{}'.format(path, text))


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


# def _reconstruct(labels, blank=0):
#     new_labels = []
#     # merge same labels
#     previous = None
#     for l in labels:
#         if l != previous:
#             new_labels.append(l)
#             previous = l
#     # delete blank
#     new_labels = [l for l in new_labels if l != blank]

#     return new_labels

# def greedy_decode(emission_log_prob, blank=0, **kwargs):
#     labels = np.argmax(emission_log_prob, axis=-1)
#     labels = _reconstruct(labels, blank=blank)
#     return labels

# def ctc_decode(log_probs, label2char=None, blank=0, method='greedy', beam_size=10):
#     emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
#     # size of emission_log_probs: (batch, length, num_classes)
    
#     decoders = {
#         'greedy': greedy_decode,
#         # 'beam_search': beam_search_decode,
#         # 'prefix_beam_search': prefix_beam_decode,
#     }
#     decoder = decoders[method]

#     decoded_list = []
#     for emission_log_prob in emission_log_probs:
#         decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
#         if label2char:
#             decoded = [label2char[l] for l in decoded]
#         decoded_list.append(decoded)
#     return decoded_list



# '''
# CHARS = [
#         '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁',
#         '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
#         #'警', '学',
#         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
#         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
#         # len ： 31 + 25 + 10 = 66
#     ]
#     # 此处将0号位置空出作为CTC-blank
#     CHARS2LABEL = {ch: i + 1 for i, ch in enumerate(CHARS)}
#     LABEL2CHAR = {label: ch for ch, label in CHARS2LABEL.items()}
# '''