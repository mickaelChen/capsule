#############################################################################
# Import                                                                    #
#############################################################################
import random
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
#############################################################################
# Hyperparameters                                                           #
#############################################################################
opt = DotDict()

opt.dataset = 'mnist'

# Input space
opt.sizeX = 28

# Hardward settings
opt.workers = 4
opt.cuda = True
opt.gpu = 0

# Optimisation scheme
opt.batchSize = 128
opt.nEpochs = 10000
opt.nRoutings = 3
opt.lambdaClfNeg = 0.5
opt.lambdaRec = 0.0005

# Load networks
opt.load = 0
opt.checkpointDir = '.'
opt.checkpointFreq = 1

#############################################################################
# Loading Weights                                                           #
#############################################################################
opt.netEnc = ''
opt.netDec = ''
if opt.load > 0:
    opt.netEnc = '%s/netEnc_%d.pth' % (opt.checkpointDir, opt.load)
    opt.netDec = '%s/netDec_%d.pth' % (opt.checkpointDir, opt.load)

#############################################################################
# RandomSeed                                                                #
#############################################################################  
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#############################################################################
# CUDA                                                                      #
#############################################################################   
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if opt.cuda:
    torch.cuda.set_device(opt.gpu)

#############################################################################
# Data-augmentation                                                         #
#############################################################################
class RandomTranslationWithPadding(object):
    def __init__(self, max_shift=2):
        self.max_shift = max_shift
    def __call__(self, pic):
        c = pic.size(0)
        h = pic.size(1)
        w = pic.size(2)
        h_shift, w_shift = np.random.randint(-self.max_shift, self.max_shift + 1, size=2)
        x = torch.FloatTensor(c, h, w).zero_()
        h_shift
        h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)

#############################################################################
# Dataset                                                                   #
#############################################################################
if opt.dataset == 'mnist':
    opt.nc = 1
    opt.nClass = 10
    dataset = dset.MNIST('/local/chenm/data/MNIST',
                         transform=transforms.Compose([transforms.Pad(2),
                                                       transforms.RandomCrop(28),
                                                       transforms.ToTensor(),
                                                      ])
                        )
    testset = dset.MNIST('/local/chenm/data/MNIST', 
                         transform=transforms.Compose([transforms.ToTensor(),
                                                      ]),
                         train=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

#############################################################################
# Modules                                                                   #
#############################################################################
def squash(s):
    square_norm_s = (s*s).sum(1).unsqueeze(1)
    v = (square_norm_s.sqrt() / (1 + square_norm_s)) * s
    return v

class _convCapsule(nn.Module):
    def __init__(self, nCapsIn, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(_convCapsule, self).__init__()
        self.conv = nn.Conv2d(in_channels=nCapsIn*in_channels,
                              out_channels=nCapsIn*out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=nCapsIn,
                              bias=bias)
        self.nCapsIn = nCapsIn
        self.out_channels = out_channels
    def forward(self, u, nRoutings=3):
        u_ = self.conv(u)
        u_ = u_.view(-1, self.nCapsIn, self.out_channels, u_.size(2), u_.size(3))
        v = squash(u_.mean(1))
        if nRoutings > 1:
            u_reshaped = u_.view(u_.size(0), u_.size(1), u_.size(2), -1).permute(0,3,1,2)
            b = 0
            for r in range(1, nRoutings):
                v_reshaped = v.view(v.size(0), v.size(1), v.size(2)*v.size(3)).permute(0,2,1).unsqueeze(-1)
                b = b + u_reshaped.matmul(v_reshaped).permute(0,2,3,1).contiguous().unsqueeze(2)
                v = squash((u_ * F.softmax(b, 1)).sum(1))
        return v

class _convCapsuleLayer(nn.Module):
    def __init__(self, nCapsIn, nCapsOut, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(_convCapsuleLayer, self).__init__()
        self.capsules = nn.ModuleList([_convCapsule(nCapsIn=nCapsIn,
                                                    in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=bias) for i in range(nCapsOut)])
    def forward(self, u, nRoutings=3):
        return torch.cat([capsule(u, nRoutings) for capsule in self.capsules],1)

#############################################################################
# Modeles                                                                   #
#############################################################################
class _encoder(nn.Module):
    def __init__(self, nf, nCaps):
        super(_encoder, self).__init__()
        self.nf = nf
        self.nCaps = nCaps
        self.layer1 = nn.Conv2d(nf[0], nf[1], 9)
        self.layer2 = nn.Conv2d(nf[1], nCaps[2] * nf[2], 9, 2)
        self.layer3 = _convCapsuleLayer(nCaps[2], nCaps[3], nf[2], nf[3], 6)
    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = squash(self.layer2(x1).view(x1.size(0), self.nCaps[2], self.nf[2], 6, 6)).view(x1.size(0), self.nCaps[2] * self.nf[2], 6, 6)
        x3 = self.layer3(x2)
        x3 = x3.view(x3.size(0), self.nCaps[3], self.nf[3])
        return x3

class _decoder(nn.Module):
    def __init__(self, nh):
        super(_decoder, self).__init__()
        self.layer1 = nn.Linear(nh[0], nh[1])
        self.layer2 = nn.Linear(nh[1], nh[2])
        self.layer3 = nn.Linear(nh[2], nh[3])
    def forward(self, x, y):
        x1 = x.masked_select(y.unsqueeze(2).expand_as(x).byte()).view(x.size(0),x.size(2))
        x2 = F.relu(self.layer1(x1))
        x3 = F.relu(self.layer2(x2))
        x4 = F.sigmoid(self.layer3(x3))
        return x4

#############################################################################
# Placeholders                                                              #
#############################################################################
x = torch.FloatTensor()
y = torch.LongTensor()   
oneHot_y = torch.FloatTensor()

#############################################################################
# Modules                                                                   #
#############################################################################
netEnc = _encoder([1, 256, 8, 16], [0, 0, 32, opt.nClass])
netDec = _decoder([16, 512, 1024, 28*28])
recCriterion = nn.MSELoss()

if opt.netDec != '':
    netDec.load_state_dict(torch.load(opt.netDec))
if opt.netEnc != '':
    netEnc.load_state_dict(torch.load(opt.netEnc))

#############################################################################
# To Cuda                                                                   #
#############################################################################
if opt.cuda:
    x = x.cuda()
    y = y.cuda()
    oneHot_y = oneHot_y.cuda()
    netEnc.cuda()
    netDec.cuda()
    recCriterion.cuda()


#############################################################################
# Optimizer                                                                 #
#############################################################################
optimizerEnc = optim.Adam(netEnc.parameters())
optimizerDec = optim.Adam(netDec.parameters())

#############################################################################
# Train                                                                     #
#############################################################################
for epoch in range(opt.load, opt.nEpochs):
    log_clf = []
    log_rec = []
    for x_cpu, y_cpu in tqdm(dataloader):
        netEnc.train()
        netDec.train()
        x.resize_(x_cpu.size(0), x_cpu.size(1), x_cpu.size(2), x_cpu.size(3)).copy_(x_cpu)
        y.resize_(y_cpu.size(0)).copy_(y_cpu)
        oneHot_y.resize_(y.size(0), opt.nClass).zero_().scatter_(1, y.unsqueeze(1), 1)
        encX = netEnc(Variable(x))
        decX = netDec(encX, Variable(oneHot_y)).view(-1,opt.nc, opt.sizeX, opt.sizeX)
        pred = (encX*encX).sum(2).sqrt()
        lossClf = (Variable(oneHot_y) * (F.relu(.9 - pred)).pow(2) + opt.lambdaClfNeg * (1 - Variable(oneHot_y)) * (F.relu(pred - .1).pow(2))).sum()
        lossRec = recCriterion(decX, Variable(x))
        (lossClf + opt.lambdaRec * lossRec).backward()
        optimizerEnc.step()
        optimizerDec.step()
        netEnc.zero_grad()
        netDec.zero_grad()
        log_clf.append(lossClf.data / y.size(0))
        log_rec.append(lossRec.data / y.size(0))
    print(epoch+1, 
          np.array(log_clf).mean(),
          np.array(log_rec).mean(),
         )
    with open('logs.dat', 'ab') as f:
        np.savetxt(f, np.vstack((np.array(log_clf),
                                 np.array(log_rec),
                                 )).T)
    if (epoch+1) % opt.checkpointFreq == 0:
        netEnc.eval()
        netDec.eval()
        acc = 0
        n = 0
        for x_cpu, y_cpu in testloader:
            netEnc.eval()
            netDec.eval()
            x.resize_(x_cpu.size(0), x_cpu.size(1), x_cpu.size(2), x_cpu.size(3)).copy_(x_cpu)
            y.resize_(y_cpu.size(0)).copy_(y_cpu)
            oneHot_y.resize_(y.size(0), opt.nClass).zero_().scatter_(1, y.unsqueeze(1), 1)
            encX = netEnc(Variable(x, volatile=True))
            decX = netDec(encX, Variable(oneHot_y, volatile=True)).view(-1,opt.nc, opt.sizeX, opt.sizeX)
            pred = (encX*encX).sum(2).sqrt()
            acc += (pred.data.max(1)[1] == y).sum()
            n += y.size(0)
        print("Accuracy: ", acc/n)
        vutils.save_image(x_cpu, 'x_%d.png' % (epoch+1), normalize=True)        
        vutils.save_image(decX.cpu().data, 'out_%d.png' % (epoch+1), normalize=True)        
        torch.save(netEnc.state_dict(), '%s/netEnc_%d.pth' % (opt.checkpointDir, (epoch+1)))
        torch.save(netDec.state_dict(), '%s/netDec_%d.pth' % (opt.checkpointDir, (epoch+1)))        
