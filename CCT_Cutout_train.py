# -*- coding: utf-8 -*-
"""TransformersCCT

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UEW8iksmdU_RFQa5eKpzEViUgyV976Jp

# Transformers Pipeline Work
"""

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.nn import Module, Linear, Dropout, LayerNorm, Identity
print('okk')

# *------------------- CCT: Transformer -------------------------*
# Meta CNN, CCT transformer
class Attention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 3, Batch , num_heads , N(seq_len) , channels // num_heads (scale)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True):
        super(Tokenizer, self).__init__()
        # 不用麻烦了，就搞一层
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]
        # input -- 中间过渡 in_planes * in_planes -- output channel

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=False),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=32, width=32):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]  # 你这，有点不优异，自己又跑了一遍可还行

    def forward(self, x):
        # return self.flattener(self.conv_layers(x)).transpose(-2, -1)
        out = self.conv_layers(x)  # 连着对 H、W downsample两次， 砍成1/4
        return self.flattener(out).transpose(-2, -1)  # [BS , downsample(H*W) , output C]

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim),
                                          requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                   requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                   requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])  # 几层 transformer
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)
        # transformers 堆叠
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class CCT(nn.Module):
    def __init__(self,
                 img_size=32,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=3,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,  # 起码三件套得整一手吧
                                   n_conv_layers=n_conv_layers)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def train(model, train_X, loss_func, optimizer):
    model.train()
    total_loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (x, y) in enumerate(train_X):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = loss_func(outputs, y)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.shape[0]
        if (i + 1) % 300 == 0:
            print("Step [{}/{}] Train Loss: {:.4f}".format(i + 1, len(train_X), loss.item()))

    print(total_loss / len(train_X.dataset))  # 其实loss计算有点小问题
    return total_loss / len(train_X.dataset)


def predict(model,test_X):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    top1_acc = top5_acc = 0
    with torch.no_grad():
        for i , (x,y) in enumerate(test_X):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, indice = output.topk(5, -1, True, sorted= True)
            top1_acc += torch.eq(indice[:, 0], y).sum().item()
            top5_acc += torch.eq(indice , y.view(-1,1)).sum().item()
    return top1_acc/len(test_X.dataset) , top5_acc/len(test_X.dataset)


"""# CUTOUT"""

# Parameters 
# 调节以选出最优！！！
CUTOUT = False
N_holes = 1
Length = 16

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes = 1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
print('okk')

"""# dataLoader

"""

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize( mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),])

# cutout or not
if CUTOUT:
    train_transform.transforms.append(Cutout(n_holes=N_holes, length=Length))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

train_dataset = torchvision.datasets.CIFAR10(root='cifar',download=True ,train=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root='cifar',download=True ,train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    normalize, ]))


# train_loader
batch_size = 128
num_workers = 2
trainLoader = torch.utils.data.DataLoader(train_dataset,
              batch_size=batch_size, shuffle=True,
              num_workers=num_workers, pin_memory=True,
              drop_last=True,) # collate_fn=collator
testLoader = torch.utils.data.DataLoader(test_dataset,
          batch_size= 512 , shuffle=False,
          num_workers=num_workers, pin_memory=True,
          drop_last=False)




print(len(trainLoader.dataset))
print(len(testLoader.dataset))

date = '621'
if CUTOUT:
  ss = 'Cutout' + str(N_holes) + '_' + str(Length) 
  signal = 'CCT7_' + date + ss + '.pth'
else:
  signal = 'CCT7_' + date + '.pth'

PATH = './drive/MyDrive/' + signal
print(PATH)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CCT(num_layers= 7,
        num_heads= 4,
        mlp_ratio= 2,
        embedding_dim= 256,
        kernel_size=3)
model.to(device)
# copy from github
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay= 5e-4,nesterov=True)
# scheduler = MultiStepLR(optimizer, milestones=[60,120,160,180], gamma=0.5)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3,weight_decay=3e-2)
# scheduler = CosineAnnealingLR(optimizer, 200, 0)

loss_func = nn.CrossEntropyLoss()
# scheduler = MultiStepLR(optimizer, milestones=[30,60,90,120,150,160,180], gamma=0.7)
print('refresh')
best_accuracy = None

def adjust_learning_rate(optimizer, epoch ):
    import math
    lr = 1e-3
    Epochs = 200
    warmup = 5

    if epoch < warmup:
        lr = lr / (warmup - epoch)
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup) / (Epochs - warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
print('okk')

"""# train"""

# train
# 三、 训练
import time
start = time.time()
Iter_Time = 200
loss_train = np.zeros(Iter_Time)
test_acc= np.zeros(Iter_Time)
test_5acc= np.zeros(Iter_Time)
train_acc = np.zeros(Iter_Time)
print('start to train' + '*'*20)
if best_accuracy is None:
  best_accuracy = 0   
for epoch in range(Iter_Time):
    adjust_learning_rate(optimizer , epoch)
    loss = train(model= model, train_X=trainLoader, loss_func=loss_func,optimizer=optimizer)
    loss_train[epoch] = loss
    top1 , top5 = predict(model=model, test_X =testLoader)
    test_acc[epoch] = top1
    test_5acc[epoch] = top5
    print('训练轮次',epoch, 'loss:' , loss ,'top1 & top5:' ,top1 , top5 , '总训练时间:', time.time() - start)
    # scheduler.step()

    if best_ accuracy < top1:
      best_accuracy = top1 
      torch.save(model.state_dict(),PATH)
      print('Model is saved in epoch_%d_accuracy_%f'%(epoch, top1 ) )
    if top1 < 0.15: break
    

print('end train now' + '*'*20)
loss_acc = np.append(loss_train , test_acc)
txtPath = './drive/MyDrive/LossAndAcc/'
np.savetxt( txtPath + signal + '.txt' , loss_acc)

print('end train now' + '*'*20)
loss_acc = np.append(loss_train , test_acc)
txtPath = './drive/MyDrive/LossAndAcc/'
np.savetxt( txtPath + signal + '.txt' , loss_acc)

