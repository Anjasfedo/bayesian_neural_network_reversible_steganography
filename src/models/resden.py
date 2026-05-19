import torch
import torch.nn as nn
import math

class trs(nn.Module):
    def __init__(self, ip, op):
        super(trs, self).__init__()
        self.ban = nn.BatchNorm2d(ip)
        self.conv = nn.Conv2d(ip, op, kernel_size=1, bias=False)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        o = self.ban(x)
        o = self.gelu(o)
        o = self.drop(o)
        o = self.conv(o)
        return o

class botn(nn.Module):
    def __init__(self, ip, k1=12, k2=12):
        super(botn, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.3)
        
        if k2 > 0:
            planes = k2 * 4
            self.bn2_1 = nn.BatchNorm2d(ip)
            self.conv2_1 = nn.Conv2d(ip, planes, kernel_size=1, bias=False)
            self.bn2_2 = nn.BatchNorm2d(planes)
            self.conv2_2 = nn.Conv2d(planes, k2, kernel_size=3, padding=1, bias=False)
            
        if k1 > 0:
            planes = k1 * 4
            self.bn1_1 = nn.BatchNorm2d(ip)
            self.conv1_1 = nn.Conv2d(ip, planes, kernel_size=1, bias=False)
            self.bn1_2 = nn.BatchNorm2d(planes)
            self.conv1_2 = nn.Conv2d(planes, k1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        if self.k2 > 0:
            ol = self.bn2_1(x)
            ol = self.gelu(ol)
            ol = self.drop(ol)
            ol = self.conv2_1(ol)
            ol = self.bn2_2(ol)
            ol = self.gelu(ol)
            ol = self.drop(ol)
            ol = self.conv2_2(ol)
            
        if self.k1 > 0:
            il = self.bn1_1(x)
            il = self.gelu(il)
            il = self.drop(il)
            il = self.conv1_1(il)
            il = self.bn1_2(il)
            il = self.gelu(il)
            il = self.drop(il)
            il = self.conv1_2(il)
            
        csize = x.size(1)
        if self.k1 == csize:
            x = x + il
        elif 0 < self.k1 < csize:
            rig = x[:, csize - self.k1: csize, :, :] + il
            lef = x[:, 0: csize - self.k1, :, :]
            x = torch.cat((lef, rig), 1)
            
        if self.k2 <= 0:
            o = x
        else:
            o = torch.cat((x, ol), 1)
            
        return o

class ResDen(nn.Module):
    def __init__(self, depth=52, unit=botn, k1=12, cr=2, k2=12):
        super(ResDen, self).__init__()
        n = (depth - 4) // 6
        self.k2 = k2
        self.k1 = k1
        self.ip = max(k2 * 2, k1)
        
        self.conv1 = nn.Conv2d(1, self.ip, kernel_size=3, padding=1, bias=False)
        self.block1 = self.make_block(unit, n)
        self.trans1 = self.make_transition(cr)
        self.block2 = self.make_block(unit, n)
        self.trans2 = self.make_transition(cr)
        self.block3 = self.make_block(unit, n)
        self.ban = nn.BatchNorm2d(self.ip)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.3)
        
        self.head_prediction = nn.Sequential(
            nn.Conv2d(self.ip, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.head_variance = nn.Conv2d(self.ip, 1, kernel_size=3, padding=1)
        self.softplus = nn.Softplus()

        for i in self.modules():
            if isinstance(i, nn.BatchNorm2d):
                i.weight.data.fill_(1)
                i.bias.data.zero_()
            elif isinstance(i, nn.Conv2d):
                nn.init.kaiming_normal_(i.weight, mode='fan_out', nonlinearity='relu')

    def make_transition(self, cr):
        op = max(int(math.floor(self.ip // cr)), self.k1)
        ip = self.ip
        self.ip = op
        return trs(ip, op)

    def make_block(self, unit, unum):
        lyrs = []
        for _ in range(unum):
            lyrs.append(unit(self.ip, k1=self.k1, k2=self.k2))
            self.ip += self.k2
        return nn.Sequential(*lyrs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.ban(x)
        x = self.gelu(x)
        x = self.drop(x)
        
        y_hat = self.head_prediction(x)
        sigma_squared = self.softplus(self.head_variance(x))
        
        return y_hat, sigma_squared