import torch
from config import config
from collections import OrderedDict
import os
from model.model import *

def generate_onnx():
    best_model_path = config.weights + config.model_name + os.sep + config.description + os.sep + str(config.fold) + os.sep + 'model_best.pth.tar'
    checkpoint = torch.load(best_model_path)

    model = get_net(config.num_classes, config.model_name)
    model.cuda()

    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        key_name = k[7:]
        new_state_dict[key_name] = v
    model.load_state_dict(new_state_dict)

    onnx_path = config.weights + config.model_name + os.sep + config.description + os.sep + str(config.fold) + os.sep + 'model_best.onnx'
    dummy_input = torch.randn(1,3,64,64, device = 'cuda')
    torch.onnx.export(model, dummy_input, onnx_path, verbose = True)

if __name__ == '__main__':
    generate_onnx()