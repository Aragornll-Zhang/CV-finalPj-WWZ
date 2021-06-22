import torch
import numpy as np

'''
对于CNN, 测试数据增强的威力
(由于Cifar-10不够优秀，测试集准确率都很高，需要额外测试一波
    加噪声法 +  卷积核失活率 + 测试集上每个layer下的激活值
@author ZXY
'''


# * ---  Method 1: 在测试集上加噪声法，再predict --- *
# e.g. 加一波高斯噪声
def AddGuassinNoise(image, mean=0, std=0.2):
    # Add noise
    N = image.flatten().shape[0]
    GussianNoise = torch.randn(N).reshape(image.shape)
    GussianNoise = GussianNoise * std + mean
    return (image + GussianNoise)


def checkNoise(model, testLoader):
    # 每个模型 predict之前,先搞随机数种子, 相同随机性
    seed = 999
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    top1_acc = top5_acc = 0
    seed = 999
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    with torch.no_grad():
        for i, (x, y) in enumerate(testLoader):
            x = AddGuassinNoise(x, mean=0, std=std)
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, indice = output.topk(5, -1, True, sorted=True)
            top1_acc += torch.eq(indice[:, 0], y).sum().item()
            top5_acc += torch.eq(indice, y.view(-1, 1)).sum().item()

    print(top1_acc / len(testLoader.dataset), top5_acc / len(testLoader.dataset))
    return


# * ---  Method 2: 在训练好的ResNet上，测试 卷积核 “失活率” --- *
def model_conv_check(model):
    '''
    广义上测试方法
        model 为 训练好的ResNet18
        检查"一枝独秀"的卷积核现象
        (也即某几个卷积核相对较大，使得其它卷积核提取的特征发挥不了威力)
    '''
    for name, paras in model.named_parameters():
        if 'conv' in name:
            print(name)
            tmp = paras.flatten(-2, -1)
            x = torch.sum(tmp * tmp, dim=-1)  # 卷积核无穷范数的平方
            x = x.flatten()
            topK, _ = torch.topk(x, k=len(x) // 10)  # 前十分之一 (可调节)
            threshold = topK[-1]
            x = x / threshold  # 相对一枝独秀 与 相对失效率 （相对值才具有统计意义, e.g. 10分之一以下
            print('失活率', torch.sum(x < 1e-4).item() / len(x))
            # this means the differece of Orders of magnitude between Top 10% and
            # the "disabled" conv kernels has achieve one percent!
    return


# *--- Method 3: 在 Cifar-10 测试集上 activation values ---*
import torch.nn as nn
from Model import *  # e.g. BasicBlock // 1*1 conv2D // 3*3 conv2D


class ResNet_ForCheck(nn.Module):
    '''
     用来测试 “ magnitude of feature activations” ( 在模型训练完之后
    '''

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_ForCheck, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.record_activations(out, idx=0)
        out = self.layer1(out)
        self.record_activations(out, idx=1)
        out = self.layer2(out)
        self.record_activations(out, idx=2)
        out = self.layer3(out)
        self.record_activations(out, idx=3)
        out = self.layer4(out)
        self.record_activations(out, idx=4)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def record_activations(self, out, idx):
        '''
            记录具体卷积过后,各个激活层的具体数值具体数值
              单独算每一张图片的激活值
        '''
        batchsize, C, H, W = out.shape
        if self.activate_record_method == 0:
            for i in range(batchsize):
                # 考虑未激活
                for channel in range(C):
                    if self.cal_method == 0:  # sum() / （H*W）
                        activation = torch.sum(out[i][channel]).item()
                    else:

                        activation = torch.sum(out[i][channel] * out[i][channel]).item()
                    self.activations[idx][channel] += activation
        else:
            # 不考虑未激活
            for i in range(batchsize):
                for channel in range(C):
                    if self.cal_method == 0:  # sum() / （H*W）
                        activation = torch.sum(out[i][channel]).item() / max(1,
                                                                             torch.sum(out[i][channel] > 1e-6).item())
                    else:
                        activation = torch.sum(out[i][channel] * out[i][channel]).item() / max(1, torch.sum(
                            out[i][channel] > 1e-6).item())
                    self.activations[idx][channel] += activation

        return

    def initial_record_activations(self, cal_method=0, activate_record_method=0):
        self.activations = [[0] * 64, [0] * 64, [0] * 128, [0] * 256, [0] * 512]
        self.cal_method = cal_method
        self.activate_record_method = activate_record_method  # 0 为 包含 Relu 后失效值 // 1 不含 Relu(x) = 0
        return


def model_checkForCifar10(model_aug, model_baseline, testLoader):
    '''
    :param model: ResNet_ForCheck model
    :param testLoader:
    :param layers: 0 -- 4
    :return:
    '''
    device = 'cpu'
    activate_record_method = 1
    model_baseline.initial_record_activations(cal_method=1, activate_record_method=activate_record_method)
    model_aug.initial_record_activations(cal_method=1, activate_record_method=activate_record_method)

    for (x, y) in testLoader:
        x = x.to(device)
        output = model_baseline(x)
        output = model_aug(x)
        break

    import matplotlib.pyplot as plt
    for layer in range(5):
        data_aug = np.array(model_aug.activations[layer])
        data_base = np.array(model_baseline.activations[layer])
        plt.plot(np.arange(len(data_aug)), data_aug / data_aug.max(), color='red')
        plt.plot(np.arange(len(data_aug)), data_base / data_base.max(), color='blue')
        plt.title('Contrast Baseline And Arugmentation in layer' + str(layer))
        # plt.savefig('cv/cv期末pj/trained_models/Contrast Baseline And Arugmentation in layer' + str(layer) + '.png')
        plt.show()
        plt.hist

    return

    Pic_size = [32, 32, 16, 8, 4]
    for idx in range(len(Pic_size)):
        size = Pic_size[idx]
        print(np.array(model_1.activations[idx]).max() / (x.shape[0]))  # size * size *

    import matplotlib.pyplot as plt

    idx = 2
    size = 16
    scale = x.shape[0] if activate_record_method else x.shape[0] * size ** 2

    return
