import os
import time
import torch
import torch.nn as nn


class IdentityModule(nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()
    def forward(self, x):
        return x


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode
    )
    fused.weight = nn.Parameter(w)
    fused.bias = nn.Parameter(b)
    return fused


def fuse_module(model):
    children = list(model.named_children())
    conv_layer = None
    conv_name = None
    for name, child in children:
        if isinstance(child, nn.BatchNorm2d) and conv_layer:
            bc = fuse(conv_layer, child)
            model._modules[conv_name] = bc
            model._modules[name] = IdentityModule()
            conv_layer = None
        elif isinstance(child, nn.Conv2d):
            conv_layer = child
            conv_name = name
        else:
            fuse_module(child)


def validate(model, input, fuse_model_path):
    model.eval()
    t0 = time.time()
    a = model(input)
    t1 = time.time()
    
    fuse_module(model)
    torch.save(model, fuse_model_path)
    t2 = time.time()
    b = model(input)
    t3 =  time.time()
    print(f"Without fusion:{t1-t0}s With fusion:{t3-t2}s")
    return (a - b).abs().max().item()

if __name__ == '__main__':
    fuse_model_path = 'results/fuse_model/best_model.pth'
    os.makedirs('results/fuse_model/', exist_ok=True)
    
    model = torch.load("logs/exp0001_batch32_epochs100_convnet/best_model.pth")
    model.to(torch.device("cuda"))
    model.eval()
    input = torch.randn(32, 3, 256, 256).cuda()
    print(validate(model, input, fuse_model_path))
