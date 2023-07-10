import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim


class TrainOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # hyper-parameters
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--epochs', type=int, default=10)
        self.parser.add_argument('--lr', type=float, default=0.0001)
        # log
        self.parser.add_argument('--log_root', type=str, default='./logs')
        # dataset
        self.parser.add_argument('--image_dir', type=str, default=None, required=True)
        self.parser.add_argument('--image_size', type=int, default=256)
        self.parser.add_argument('--shuffle', type=bool, default=True)
        # model
        self.parser.add_argument('--model', type=str, choices=['convnet', 'vggnet', 'resnet', 'densenet'], default='convnet')
        self.parser.add_argument('--gpu', type=int, default=0)
        # train
        self.parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam')
        self.parser.add_argument('--criterion', type=str, choices=['ce', 'mse'], default='ce')
    
    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.device = torch.device(f'cuda:{self.opt.gpu}')
        
        completed_exp = os.listdir(self.opt.log_root)
        if not len(completed_exp):
            self.opt.exp_num = 1
        else:
            self.opt.exp_num = max([int(dir[3:7]) for dir in completed_exp]) + 1
        self.opt.log_dir = os.path.join(self.opt.log_root, 
                                        f'exp{self.opt.exp_num:04d}_batch{self.opt.batch_size}_epochs{self.opt.epochs}_{self.opt.model}')
        self.opt.model_dir = os.path.join(self.opt.log_dir, 'best_model.pth')
        
        if self.opt.criterion == 'ce':
            self.opt.criterion = nn.CrossEntropyLoss()
        elif self.opt.criterion == 'mse':
            self.opt.criterion = nn.MSELoss()
        
        if self.opt.optimizer == 'adam':
            self.opt.optimizer = optim.Adam
        elif self.opt.optimizer == 'sgd':
            self.opt.optimizer = optim.SGD
        return self.opt


class InferenceOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # hyper-parameters
        self.parser.add_argument('--batch_size', type=int, default=32)
        # dataset
        self.parser.add_argument('--image_dir', type=str, default=None, required=True)
        self.parser.add_argument('--image_size', type=int, default=256)
        self.parser.add_argument('--shuffle', type=bool, default=True)
        # model
        self.parser.add_argument('--pretrained_model', type=str, default='./pretrained_models/conv_net.pth')
        self.parser.add_argument('--gpu', type=int, default=0)
        # train
        self.parser.add_argument('--criterion', type=str, choices=['ce', 'mse'], default='ce')
    
    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.device = torch.device(f'cuda:{self.opt.gpu}')
        if self.opt.criterion == 'ce':
            self.opt.criterion = nn.CrossEntropyLoss()
        elif self.opt.criterion == 'mse':
            self.opt.criterion = nn.MSELoss()
        return self.opt


class VisualizeOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # model
        self.parser.add_argument('--pretrained_model', type=str, default='./pretrained_models/conv_net.pth')
        self.parser.add_argument('--gpu', type=int, default=0)
        # data
        self.parser.add_argument('--image_dir', type=str, default=None, required=True)
        self.parser.add_argument('--image_size', type=int, default=256)
        # visualize
        self.parser.add_argument('--save_dir', type=str, default='./results')
        self.parser.add_argument('--save_name', type=str, default=None)
        self.parser.add_argument('--layer', type=str, default=None, help='target layer for CAM visualization')
        self.parser.add_argument('--seed', type=int, default=42)
    
    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.device = torch.device(f'cuda:{self.opt.gpu}')
        os.makedirs(self.opt.save_dir, exist_ok=True)
        if not self.opt.save_name:
            if self.opt.layer:
                self.opt.save_name = os.path.basename(self.opt.pretrained_model)[: -4] + f'_{self.opt.layer}.png'
            else:
                self.opt.save_name = os.path.basename(self.opt.pretrained_model)[: -4] + '.png'
        return self.opt
