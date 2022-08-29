#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author :          lance
@Email :      wangyl306@163.com
@Time  :   2022/07/07 11:29:49
"""

import argparse
import os

import torch
from timm import create_model
from nets.cnn_ctc import CNN
import torch.nn as nn

pretrain_path = './checkpoint/ckpoint2/cnn_156000_loss0.162173.pth'
class ModelHUb:
    def __init__(self, opt, model, num_class=1000):
        self.model_name = opt.model_name
        self.pretrained_weights = opt.pretrained_weights
        self.convert_mode = opt.convert_mode
        self.num_class = num_class
        self.img = torch.randn(1, 3, 32, 100)   #opt.size

        if os.path.exists(opt.save_dir) is False:
            os.makedirs(opt.save_dir)

        self.save_file = os.path.join(opt.save_dir, self.model_name + "_" + opt.model_library + "." + self.convert_mode)

        self.model = model
        self.model.eval()

        if hasattr(model, 'repvgg'):
            self.model.repvgg.switch_to_deploy()


    def get_model(self):
        if self.convert_mode == "onnx":
            print('+++++++++++++++++++')
            torch.onnx.export(self.model, self.img, self.save_file, input_names=["input"], opset_version=10)
        else:
            self.model(self.img)  # dry runs
            scripted_model = torch.jit.trace(self.model, self.img, strict=False)
            torch.jit.save(scripted_model, self.save_file + '.pt')
        print("[INFO] convert model save:", self.save_file)


parse = argparse.ArgumentParser(description="MAKE MODELS CLS FOR VACC")
parse.add_argument("--model_library", type=str, default="timm", choices=["timm", "torchvision"])
parse.add_argument("--model_name", type=str, default="resnet18_ocr2")
parse.add_argument("--save_dir", type=str, default="./work_dir")
# parse.add_argument("--size", type=int, default=224)  #输入大小   已经resize到224了
parse.add_argument(
    "--pretrained_weights", type=str, default=pretrain_path, help="timm or torchvision or custom onnx weights path",
)
parse.add_argument(
    "--convert_mode", type=str, default="onnx", choices=["onnx", "torchscript"],
)
args = parse.parse_args()
print(args)

if __name__ == "__main__":
    # model = create_model(model_name='resnet18', num_classes=67, pretrained=False)
    model = CNN(input_c=3, input_h=32, num_classes=66 + 1)

    
    # sta = torch.load(pretrain_path, map_location='cpu')
    # model.load_state_dict(sta['state_dict'])
    maker = ModelHUb(args, model)
    maker.get_model()
    print()
