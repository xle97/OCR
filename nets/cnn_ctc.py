import torch
from torch import nn, softmax
from pytorch_model_summary import summary
from torchvision import models
from torch.nn.functional import log_softmax


class CNN(nn.Module):   ###使用resnet18
    # 输入维度为 C * H * W = (1,32,W)
    def __init__(self, input_c, input_h, num_classes, leaky_relu=False):
        super(CNN, self).__init__()
        assert input_h % 16 == 0, 'input_h has to be a multiple of 16'
        self.input_c = input_c
        self.input_h = input_h
        self.leaky_relu = leaky_relu
        self.num_class=num_classes
        # self.cnn = self.cnn_module()
        self.cnn = nn.Sequential(*list(models.resnet18().children())[0:-3])
        self.fc = nn.Linear(256,67)
        

    def forward(self, x):
        net=self.cnn
        # cls=self.num_class

        conv_features = net(x)

        B, C, H, W = conv_features.size()
        
        conv_features=conv_features.reshape(B,C,H*W)

        conv_features = conv_features.permute(2, 0, 1)    # [W,B,C]
        # return conv_features
        out=self.fc(conv_features)

        return out
       


if __name__ == '__main__':
    net = CNN(input_c=3, input_h=32, num_classes=1000)   #input channel 改为3测试


    print(summary(net, torch.zeros(5, 1, 32, 100)))
