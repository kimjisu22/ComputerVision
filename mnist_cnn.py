import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import time
from Own_ML_Class import CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
f_mnist_train = dsets.FashionMNIST(root='./data', train=True,
                       transform=transforms.ToTensor(), download=True)
mnist_train = dsets.MNIST(root='./data', train=True,
                          transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='./data', train=False,
                         transform=transforms.ToTensor(), download=True)

#Dataloader
data_loader = torch.utils.data.DataLoader(mnist_train,
                                          batch_size=100, shuffle=True) #찾아보기. 배치사이즈 크게 잡을수록 속도는 빠르다.
def train(epoch):
    for data, gt in data_loader:
        # data shape: 28x28 --> nn.Linear의 입력값과 동이랗게 맞춰줘야 함.
        data = data.view(-1, 28*28)
        data = data.to(device)
        gt = gt.to(device)
        optimizer.zero_grad()
        y_hat = model(data)
        loss = loss_fn(y_hat, gt)
        loss.backward()
        optimizer.step()
    print(f'epoch {epoch}, loss: {loss.item()}')

# CNN 모델 정의 등
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss() # softmax(wx+b)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(4):
    # batch_size=100, training dataset: 50000
    # iter: 50000/100 -> 500
    avg_loss = 0
    for X,Y in data_loader:

        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        Y_hat = model(X)
        loss = loss_fn(Y_hat, Y)
        loss.backward() # backpropagation(미분)
        optimizer.step() # updates wights
    print(f'epoch:{epoch}, loss: {loss.item()}')


    torch.save(model, 'mnist_cnn.pth') # 모델 아키텍처 + 파라미터 저장
    torch.save(model.state_dict(), 'mnist_cnn.pt') # 둘 중 하나는 파라미터만 저장

# 모델 로드
# 잘 맞추는지 테스트
# Accuracy 확인











