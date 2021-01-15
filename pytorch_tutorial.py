import torch, code,copy
import torchvision
import torchvision.transforms as transforms
from cnn_finetune import make_model
# from cifar_dataset import cifar10
# 이미지 뭉치 형태로 dataset을 받으면, init, len, getitem
from cifar_dataset import cifar10 
# type(device) -> class 'torch.device'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)

# 이미지 변형하기
# torchvision.transforms.Compose(transforms) 
    # -> 여러개의 데이터 transformer을 묶어놓을 수 있다
transform = transforms.Compose(
    # 이미지 데이터를 tensor로 바꿔준다.
    [transforms.ToTensor(),
    # transforms.Normalize(mean, std, inplace=False) - 이미지를 정규화한다.
    # Image 한장은 R,G,B의 3채널로 각 채널마다 [0,255]의 값으로 매핑시킬 수 있을 것이다. 
    # torchvision 은 이러한 Image를 [0,1] 범위를 갖도록 출력해주고, 
    # 우리는 이를 [-1,1]로 한번 더 정규화 하는 것이다.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# cifar10_dataset과 연결
# path가 cifar -> cifar 폴더에 들어간다는 의미
# cifar_dataset 코드를 받아서 cifar10을 import 시킴 -> 사용
trainset = cifar10('cifar', transform, True)
# 다운받은 데이터셋 불러오기
# 다운 받은 데이터셋은 DataLoader 함수를 이용해야 모델에 사용할 수 있다.
# 데이터셋을 배치별로 자르거나, 섞거나 해서 모델에 넣을 데이터 뭉치를 만들어준다.
"""
#파라미터
batch_size : 모델을 한 번 학습시킬 때 몇 개의 데이터를 넣을지 정한다. 1 배치가 끝날때마다 파라미터를 조정한다.
shuffle : 데이터를 섞을지 정한다.
num_workers : 몇개의 subprocesses를 가동시킬건지 정한다.
drop_last : 배치별로 묶고 남은 데이터를 버릴지 (True) 여부를 정한다.
"""
trainloader = torch.utils.data.DataLoader(trainset,
                batch_size = 4, shuffle = True, num_workers = 0)

testset = cifar10('cifar', transform, False)
testloader = torch.utils.data.DataLoader(testset,
                batch_size = 4, shuffle = True, num_workers = 0)
"""
# 불러온 DataLoader는 데이터셋을 batch로 나누기 위해서 사용

***Batch Size***
 - 한 Step이 일어날 때 Parallel하게 처리되는 Data의 수
 - 공장이 한 차례 돌아갈 때, 라인 8개가 동시에 돌아가는 것을 떠올리면 쉽다
 - 모델의 weight를 한번 업데이트 시킬 때 사용되는 샘플들의 묶음을 의미

***Epoch***
 - 모든 데이터 셋을 한번 학습

데이터 총 100개이고 batch size가 10이면,
1 epoch에서 10개의 dataset이 한번에 처리되고, 하나의 dataset에 10개의 데이터가
포함되어 있게 됨 -> 1 epoch당 10번의 weight가 업데이트 (10 iteration)

***Iteraion***
1 epoch를 마치는데 필요한 미니배치 갯수를 의미함, epoch를 나누어서 실행하는 횟수

!!***정리***!!
전체 2000 개의 데이터가 있고, epochs = 20, batch_size = 500이라고 가정합시다.
그렇다면 1 epoch는 각 데이터의 size가 500인 batch가 들어간 네 번의 iteration으로 나누어집니다.
그리고 전체 데이터셋에 대해서는 20 번의 학습이 이루어졌으며, iteration 기준으로 보자면 총 80 번의 학습이 이루어진 것입니다.

>>> type(trainloader)
<class 'torch.utils.data.dataloader.DataLoader'>
>>> len(trainloader)
12500
!! batch_size를 4로 설정 -> 총 cifar10의 데이터 개수가 50000개 
-> 1개의 batch에 4개의 데이터가 들어가 있으니 총 batch 개수는 12500개
(한번에 4개의 데이터를 처리할 수 있는 능력을 가짐)

#데이터셋 안의 이미지 개별 확인
>>> dataiter = iter(trainloader)
>>> images, labels = dataiter.next()
# 4차원 텐서
# 이미지 배치에 많이 사용되며, 병렬처리 특히 여러 장의 이미지를 배치로 묶어 처리할 때 사용
>>> images.size()
torch.Size([4, 3, 32, 32])      # [batch값, 채널, 이미지 너비, 높이]
>>> labels
tensor([6, 9, 7, 7])        # 가져온 데이터셋 안의 이미지가 4개인데, 각각에 해당하는 class의 정답 라벨
>>> labels.size()
torch.Size([4])

>>> len(trainset)
50000
>>> len(testset)
10000

>>> type(testset)
<class 'cifar_dataset.cifar10'>
>>> type(trainset[0])
<class 'tuple'>
>>> type(trainset[0][0])
<class 'torch.Tensor'>
>>> type(trainset[0][1])
<class 'int'>

# 3차원 텐서
# 주로 이미지와 같은 형태의 데이터를 표현할 때 사용
# 여러 행렬의 층으로 쌓여있는 형태로 표현됨
>>> trainset[0][0].shape
torch.Size([3, 32, 32])     # 3채널(RGB), 8*8 사이즈의 이미지임을 확인
>>> testset[0][0].shape
torch.Size([3, 32, 32])

>>> trainset[0]
(tensor([[[-0.5059, -0.4980, -0.4902,  ..., -0.3490, -0.3569, -0.3569],
         [-0.5059, -0.4980, -0.4824,  ..., -0.3490, -0.3569, -0.3569],
         [-0.4980, -0.4902, -0.4745,  ..., -0.3490, -0.3490, -0.3569],
         ...,
         [ 0.6157,  0.6000,  0.6157,  ...,  0.1922,  0.3412,  0.4118],
         [ 0.7098,  0.7020,  0.7098,  ...,  0.6549,  0.6863,  0.6941],
         [ 0.7882,  0.7804,  0.7882,  ...,  0.7569,  0.7490,  0.7490]],

        [[-0.4980, -0.4902, -0.4824,  ..., -0.3569, -0.3647, -0.3647],
         [-0.4980, -0.4902, -0.4745,  ..., -0.3569, -0.3647, -0.3647],
         [-0.4902, -0.4824, -0.4667,  ..., -0.3569, -0.3569, -0.3647],
         ...,
         [ 0.6314,  0.6157,  0.6314,  ...,  0.2078,  0.3569,  0.4275],
         [ 0.7255,  0.7176,  0.7255,  ...,  0.6706,  0.7020,  0.7098],
         [ 0.8039,  0.7961,  0.8039,  ...,  0.7725,  0.7647,  0.7647]],

        [[-0.5373, -0.5294, -0.5216,  ..., -0.3725, -0.3804, -0.3804],
         [-0.5373, -0.5294, -0.5137,  ..., -0.3725, -0.3804, -0.3804],
         [-0.5294, -0.5216, -0.5059,  ..., -0.3725, -0.3725, -0.3804],
         ...,
         [ 0.6235,  0.6078,  0.6235,  ...,  0.2000,  0.3490,  0.4196],
         [ 0.7176,  0.7098,  0.7176,  ...,  0.6627,  0.6941,  0.7020],
         [ 0.7961,  0.7882,  0.7961,  ...,  0.7647,  0.7569,  0.7569]]]), 4)
>>> len(trainset[0])
2

>>> testset[0]
(tensor([[[0.2627, 0.2784, 0.2706,  ..., 0.3020, 0.3098, 0.2784],
         [0.2549, 0.2784, 0.2706,  ..., 0.2941, 0.3020, 0.2706],
         [0.2627, 0.2863, 0.2706,  ..., 0.2863, 0.2941, 0.2627],
         ...,
         [0.2392, 0.2627, 0.2627,  ..., 0.2549, 0.2627, 0.2392],
         [0.2314, 0.2549, 0.2392,  ..., 0.2471, 0.2627, 0.2392],
         [0.2235, 0.2471, 0.2471,  ..., 0.2471, 0.2471, 0.2235]],

        [[0.6784, 0.7020, 0.6863,  ..., 0.6941, 0.7098, 0.6863],
         [0.6706, 0.6941, 0.6784,  ..., 0.6863, 0.7020, 0.6784],
         [0.6784, 0.7020, 0.6863,  ..., 0.6941, 0.7098, 0.6863],
         ...,
         [0.6314, 0.6549, 0.6392,  ..., 0.6392, 0.6549, 0.6235],
         [0.6235, 0.6471, 0.6392,  ..., 0.6314, 0.6471, 0.6157],
         [0.6157, 0.6392, 0.6235,  ..., 0.6235, 0.6392, 0.6157]],

        [[0.9294, 0.9529, 0.9373,  ..., 0.9529, 0.9686, 0.9294],
         [0.9137, 0.9451, 0.9294,  ..., 0.9451, 0.9608, 0.9216],
         [0.9294, 0.9529, 0.9373,  ..., 0.9451, 0.9608, 0.9294],
         ...,
         [0.8902, 0.9137, 0.9059,  ..., 0.8980, 0.9137, 0.8745],
         [0.8824, 0.9059, 0.8902,  ..., 0.8902, 0.9059, 0.8745],
         [0.8667, 0.8980, 0.8824,  ..., 0.8824, 0.8980, 0.8667]]]), 2)

>>> trainset[0][0]
tensor([[[-0.5059, -0.4980, -0.4902,  ..., -0.3490, -0.3569, -0.3569],
         [-0.5059, -0.4980, -0.4824,  ..., -0.3490, -0.3569, -0.3569],
         [-0.4980, -0.4902, -0.4745,  ..., -0.3490, -0.3490, -0.3569],
         ...,
         [ 0.6157,  0.6000,  0.6157,  ...,  0.1922,  0.3412,  0.4118],
         [ 0.7098,  0.7020,  0.7098,  ...,  0.6549,  0.6863,  0.6941],
         [ 0.7882,  0.7804,  0.7882,  ...,  0.7569,  0.7490,  0.7490]],
# ---------------------------------1set--------------------------------
        [[-0.4980, -0.4902, -0.4824,  ..., -0.3569, -0.3647, -0.3647],
         [-0.4980, -0.4902, -0.4745,  ..., -0.3569, -0.3647, -0.3647],
         [-0.4902, -0.4824, -0.4667,  ..., -0.3569, -0.3569, -0.3647],
         ...,
         [ 0.6314,  0.6157,  0.6314,  ...,  0.2078,  0.3569,  0.4275],
         [ 0.7255,  0.7176,  0.7255,  ...,  0.6706,  0.7020,  0.7098],
         [ 0.8039,  0.7961,  0.8039,  ...,  0.7725,  0.7647,  0.7647]],
# ---------------------------------1set--------------------------------
        [[-0.5373, -0.5294, -0.5216,  ..., -0.3725, -0.3804, -0.3804],
         [-0.5373, -0.5294, -0.5137,  ..., -0.3725, -0.3804, -0.3804],
         [-0.5294, -0.5216, -0.5059,  ..., -0.3725, -0.3725, -0.3804],
         ...,
         [ 0.6235,  0.6078,  0.6235,  ...,  0.2000,  0.3490,  0.4196],
         [ 0.7176,  0.7098,  0.7176,  ...,  0.6627,  0.6941,  0.7020],
         [ 0.7961,  0.7882,  0.7961,  ...,  0.7647,  0.7569,  0.7569]]])
>>> len(trainset[0][0])
3 -> 총 3 세트

>>> trainset[0][1]
4
>>> testset[0][1]
2

>>> trainset[0][0][0]
tensor([[-0.5059, -0.4980, -0.4902,  ..., -0.3490, -0.3569, -0.3569],
        [-0.5059, -0.4980, -0.4824,  ..., -0.3490, -0.3569, -0.3569],
        [-0.4980, -0.4902, -0.4745,  ..., -0.3490, -0.3490, -0.3569],
        ...,
        [ 0.6157,  0.6000,  0.6157,  ...,  0.1922,  0.3412,  0.4118],
        [ 0.7098,  0.7020,  0.7098,  ...,  0.6549,  0.6863,  0.6941],
        [ 0.7882,  0.7804,  0.7882,  ...,  0.7569,  0.7490,  0.7490]])
>>> len(trainset[0][0][0])
32

>>> trainset[0][0][1]
tensor([[-0.4980, -0.4902, -0.4824,  ..., -0.3569, -0.3647, -0.3647],
        [-0.4980, -0.4902, -0.4745,  ..., -0.3569, -0.3647, -0.3647],
        [-0.4902, -0.4824, -0.4667,  ..., -0.3569, -0.3569, -0.3647],
        ...,
        [ 0.6314,  0.6157,  0.6314,  ...,  0.2078,  0.3569,  0.4275],
        [ 0.7255,  0.7176,  0.7255,  ...,  0.6706,  0.7020,  0.7098],
        [ 0.8039,  0.7961,  0.8039,  ...,  0.7725,  0.7647,  0.7647]])
>>> trainset[0][0][0][0]
tensor([-0.5059, -0.4980, -0.4902, -0.4745, -0.4667, -0.4588, -0.4431, -0.4431,
        -0.4275, -0.4275, -0.3804, -0.2078, -0.3333, -0.4039, -0.2941, -0.3255,
        -0.3804, -0.3725, -0.3725, -0.3569, -0.3569, -0.3569, -0.3647, -0.3647,
        -0.3569, -0.3490, -0.3490, -0.3490, -0.3490, -0.3490, -0.3569, -0.3569])
>>> len(trainset[0][0][0][0])
32

>>> trainset[0][0][0][0][0]
tensor(-0.5059)
>>> type(trainset[0][0][0][0][0])
<class 'torch.Tensor'>

"""

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.imshow(npimg, (1, 2, 0))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
#plt.savefig('temp.png')
#plt.savefig()
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 뉴럴 네트워크 라이브러리 nn과 nn.functional을 불러 오자. 
# nn 은 뉴럴 네트워크 모델을 지원하며 
# nn.functional은 activation을 담당하는 Relu 함수처리를 지원한다.
# nn의 경우 레이어 연산에서 weigh 공유가 가능.
# F의 경우 단순한 연산만 가능
import torch.nn as nn
import torch.nn.functional as F

# 합성곱 신경망(Convolution Neural Network) 정의하기
# 사용된 모델은 Lenet5
# 클라스 Net 의 initialization 과정에서 외부의 Parent 클라스 대신 
# Net 자체를 super(Net, self)를 사용하여 처리하면서 초기화
# 입력 Variable을 받아 다른 모듈 또는 Variable의 autograd 연산을 사용하여 출력 Variable을 생성 하는 forward 정의
# 연산 그래프와 autograd는 복잡한 연산자를 정의하고 도함수(derivative)를 자동으로 계산
# 모듈은 입력 Variable을 받고 출력 Variable을 계산

class Net(nn.Module):
    #1 모델에 필요한 연산 정의
    def __init__(self):
        super(Net, self).__init__()
        # Convolution 층 생성
        self.conv1 = nn.Conv2d(3, 6, 5)     # input channels, output channels, kernel size
        # Maxpooling 층 생성
        self.pool = nn.MaxPool2d(2, 2)      # kernel size, stride, padding = 0 (default)
        # Convolution 층 생성
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connected 층 생성
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # input features, output features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    #2 모델에서 연산이 이루어지는 순서 설정
    # instance 계산을 위해 입력 데이터 x 가 주어지면 
    # method인 forward 가 계산 후 최종 값을 돌려준다.
    def forward(self, x):
        # 이미지를 conv1~2 - relu - pool 순서로 처리
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # fully connected 층에 넣기 위해 이미지를 펴주는 작업
        x = x.view(-1, 16 * 5 * 5)
        # 이미지를 fc1~2 - relu 순서로 처리
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 최종 순서로, 이전의 층을 거쳐온 결과값을 fc3에 입력
        x = self.fc3(x)
        return x

#3 모델을 특정 인스턴스로 설정하기
net = Net()

print(net)

"""
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
"""

model_path = "model_saved.pth"
torch.save(net.state_dict(), model_path)
# 네트워크 로딩
net.load_state_dict(torch.load(model_path))
net.to(device)
net.train()

# transfer learing, cnn_finetune의 make_model 함수를 사용
# pretrained -> 학습된 파라미터 데려온다
# input size도 지정해야 함, vgg의 default를 따르지 않을 수도 있기 때문
model_pretrained = make_model('vgg16', num_classes=10, pretrained=True, input_size=(32, 32))
model_original = make_model('vgg16', num_classes=10, pretrained=False, input_size=(32, 32))

"""
>>> model_pretrained
VGGWrapper(
  (_features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (_classifier): Sequential(
    (0): Linear(in_features=512, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=10, bias=True)
  )
)

>>> model_original
VGGWrapper(
  (_features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (_classifier): Sequential(
    (0): Linear(in_features=512, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=10, bias=True)
  )
)
"""

# 마지막 레이어 사이즈 다를 수 있기 때문에 데려오지 않을 것
not_copy = ['_classifier.0.weight']
model_original_dict = model_original.state_dict()

"""
>>> model_original_dict.keys()
odict_keys(['_features.0.weight', '_features.0.bias', '_features.2.weight', 
'_features.2.bias', '_features.5.weight', '_features.5.bias', '_features.7.weight', 
'_features.7.bias', '_features.10.weight', '_features.10.bias', '_features.12.weight', 
'_features.12.bias', '_features.14.weight', '_features.14.bias', '_features.17.weight', 
'_features.17.bias', '_features.19.weight', '_features.19.bias', 
'_features.21.weight', '_features.21.bias', '_features.24.weight', 
'_features.24.bias', '_features.26.weight', '_features.26.bias', 
'_features.28.weight', '_features.28.bias', '_classifier.0.weight', 
'_classifier.0.bias', '_classifier.3.weight', '_classifier.3.bias', 
'_classifier.6.weight', '_classifier.6.bias'])

"""

# layer 이름, parameter 값 = key, value
pretrained_state = {key:value for key,value in 
								model_pretrained.state_dict().items() 
								if key not in not_copy}

"""
# 업데이트 전
>>> list(model_original_dict.values())[-1]
tensor([-0.0017,  0.0112, -0.0086,  0.0128,  0.0049, -0.0062,  0.0119, -0.0004,
        -0.0022, -0.0084])
>>> list(pretrained_state.values())[-1]
tensor([ 0.0060,  0.0076,  0.0118,  0.0037, -0.0005,  0.0024,  0.0002, -0.0137,
         0.0024, -0.0096])
"""
#code.interact(local = dict(globals(),**locals()))


# not copy외 모두 업데이트
model_original_dict.update(pretrained_state)
model_original.load_state_dict(model_original_dict)

"""
# 업데이트 후
>>> list(model_original_dict.values())[-1]
tensor([-0.0068, -0.0078,  0.0087,  0.0044,  0.0110,  0.0013, -0.0015, -0.0132,
         0.0028, -0.0022])
>>> list(pretrained_state.values())[-1]
tensor([-0.0068, -0.0078,  0.0087,  0.0044,  0.0110,  0.0013, -0.0015, -0.0132,
         0.0028, -0.0022])
"""

#net = make_model('resnet18', num_classes=10, pretrained=True,, input_size=(32, 32))


net.to(device)
#torch.optim 라이브러리
#loss function과 grad를 조절해주는 optimizer를 정의
# optimizer를 통해 loss 값을 줄이는 방향으로 파라미터들을 조정
import torch.optim as optim

# criterion : 손실함수 지정
# nn.CrossEntropyLoss()의 경우 기본적으로 LogSoftmax()가 내장
# 'CrossEntropyLoss'는 파이토치가 제공하는 대표적인 loss function으로, 다중 분류에 사용
criterion = nn.CrossEntropyLoss()
# optimizer : 최적화 함수 지정 :
# model.parameters()를 통해 model의 파라미터들을 할당.
# lr : learnig_rate 지정

# Stochastic Gradient Descent(SGD) 옵티마이저를 설정하는데 
# back-propagation알고리즘과 관련된 net.parameters( )의
# learning rate 값과 모멘텀 값을 설정
# -> loss가 최소가 되는 지점 찾고, 이 때의 weight 값 찾기
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#training set을 통해 모델 실행하기

#1 데이터를 몇 번 돌릴 것인가? -> 2회
for epoch in range(2):  # loop over the dataset multiple times

# loss.backward() 명령에 따른 Back-propagation 루프 과정에서 
# loss 함수 계산을 위해 사용하는 웨이트 좌표 값들 위치에서 
# loss 함수의 웨이트 변수에 대한 편미분 값 
# 즉 gradient값을 계산하여 learning rate를 곱하여 뺀 새 웨이트 좌표 값에 대해 
# loss 함수를 계산해 나가면서 최소값을 찾아내게 된다. 

# 따라서 매번 계산했던 gradient 값을 optimizer.zero_grad() 명령을 사용하여 
# 0.0으로 두고 새 웨이트 좌표 값에 대해 loss 함수 값을 계산한다.
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):   # trainloader에서 학습 데이터를 불러옴 (훈련용 데이터)
        # get the inputs
        inputs, labels = data   # 학습 데이터에는 이미지와 레이블이 존재, 두가지를 data로 설정
        inputs, labels = inputs.to(device), labels.to(device)
    
        # zero the parameter gradients
        optimizer.zero_grad()   # optimizer를 초기화시켜주는 작업

        #2 forward + backward + optimize 수행
        outputs = net(inputs)   # 데이터(inputs)를 모델에 넣어 나온 결과값이 output
        loss = criterion(outputs, labels)   # output과 레이블의 loss 값 계산(CrossEntropyLoss를 기준으로_criterion)
        loss.backward()     # loss를 기준으로 backward
        # 마지막의 optimizer.step()는 한 번의 back-propagation 알고리즘에 
        # 의해서 loss 함수를 계산한 후 다음 스텝으로 넘어가기 위한 준비를 뜻한다.
        optimizer.step()    # optimize 진행

        #3 누적된 loss를 계산
        # 2000번째 batch마다 누적 loss 값의 중간 결과를 출력, print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0      # loss가 0이 되면 학습 종료

print('Finished Training')
code.interact(local = dict(globals(),**locals()))

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images with 시험용 데이터
# batch용 테스트 데이터 중에서 무작위호 batch_size=4에 해당하는 
# 한 묶음 데이터 4개를 램덤하게 불러내어 그래픽 출력 확인
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Image 한장이 입력 되었을 때, 출력은 각 10개 Class에 대한 값으로 나타난다. 
# 어떤 Class에 대해서 더 높은 값이 나타난다는 것은, 
# 모델(신경망)이 입력 이미지는 해당 Class에 더 가깝다고 생각한다는 것이다. 
# 따라서, 가장 높은 값을 갖는 인덱스(index)를 뽑으면 그것이 훈련된 모델의 예측 결과이다.

# 앞 단계에서 학습한 결과를 사용하여 
# 임의로 추출한 4개의 이미지 데이터에 대한 instance를 계산
outputs = net(images.cuda())

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# 훈련 데이터셋을 모두 사용한 뒤에는 제대로 학습이 되었는지 확인하기 위해 테스트를 진행
# 신경망이 예측한 정답과 진짜 정답(Ground-truth)을 비교하는 방식으로 확인하고, 
# 만약 예측이 맞다면 샘플을 ‘맞은 예측값(Correct predictions)’에 넣도록

correct = 0
total = 0
# 테스트에서는 모델을 실행할 때 업데이트가 필요 없기 때문
with torch.no_grad():   # grad를 업데이트 안한다는 의미
    #  testset도 testloader를 설정했었음
    for data in testloader:     # test 데이터도 batch이기 때문에 for문 필요
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)   # 10개 레이블 중 가장 큰 값으로 1 개의 레이블을 결과값으로
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# classifier의 정확도에 대해 출력
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# 모델이 어떤 class에서 잘 분류하고, 어떤 class에서 잘 분류하지 못하는지
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))