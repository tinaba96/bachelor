# freda (todo) : 

import torch.nn as nn
import torch
import numpy as np 

class QtActive(torch.autograd.Function):
    '''
    Quantize the input activations and calculate the mean across channel dimension
.
    '''
    def __init__(self, bw=8):
        super(QtActive, self).__init__()
        self.lev = pow(2., int(bw))           #quantization levels
        self.max = (self.lev - 1.) / self.lev #maximum number

    def forward(self, input):
        self.save_for_backward(input)
        tmp = input.data.clamp(0.0, self.max)
        input.data = tmp.mul(self.lev).add(0.5).floor().div(self.lev)
        return input

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.le(0.0)] = 0.0
        grad_input[input.ge(self.max)] = 0.0
        return grad_input


class QtConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=3, stride=1, padding=1, dropout=0, bw=8):

        super(QtConv2d, self).__init__()
        self.layer_type = 'QtConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(output_channels, eps=1e-4, momentum=0.1, affine=True)
        self.qa = QtActive(bw)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        x = 0.25 * x #quantize values in the range of 4 sigma
        x = self.qa(x)
        x = self.relu(x)
        return x


def qconv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        QtConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2),
    )

def qdeconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        QtConv2d(in_planes,out_planes,kernel_size=3,stride=1,padding=1)
    )

def cbr(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

def ccbr(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias = True):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
            nn.BatchNorm2d(out_planes),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
        )

def predict_flow(in_planes, out_planes=2):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

#def deconv(in_planes, out_planes):
#    return nn.Sequential(
#        nn.Upsample(scale_factor=2, mode='bilinear'),
#        nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=1,padding=1)
#    )

class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def init_deconv_bilinear(weight):
    f_shape = weight.size()
    heigh, width = f_shape[-2], f_shape[-1]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([heigh, width])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weight.data.fill_(0.)
    for i in range(f_shape[0]):
        for j in range(f_shape[1]):
            weight.data[i,j,:,:] = torch.from_numpy(bilinear)


def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook

'''
def save_grad(grads, name):
    def hook(grad):
        grads[name] = grad
    return hook
import torch
from channelnorm_package.modules.channelnorm import ChannelNorm 
model = ChannelNorm().cuda()
grads = {}
a = 100*torch.autograd.Variable(torch.randn((1,3,5,5)).cuda(), requires_grad=True)
a.register_hook(save_grad(grads, 'a'))
b = model(a)
y = torch.mean(b)
y.backward()

'''



