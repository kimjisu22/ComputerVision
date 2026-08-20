import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as dsets            # mnist, fashin mnist 등 가지고 올 수 있음
import torchvision.transforms as transforms     # dataset --> Tensor로 변경, to Image 이미지로 변경을 하거나... 그런 기능들이 있음
                                                # dataset이 다운을 받으면 g?z?파일로 존재를 하는데 그거를 텐서나 이미지로 변환을 해줌...
import time

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
f_data_loader = torch.utils.data.DataLoader(f_mnist_train,
                                          batch_size=100, shuffle=True)
transform = transforms.ToPILImage()
# print(mnist_train)
# for i in range(5):
#     img = transform(mnist_train[i][0])
#     print(mnist_train[i][1])
#     print(img.size)
#     plt.imshow(img, cmap='gray')
#     plt.show()
###################################################################################
# model(x) = w*x +b <-  객체 생성 <- model에 x값 넣으면 저렇게 계싼해주는 녀석임
# (28, 28) --> 2차원 데이터로 표현 --> 1차원으로 계산하기 위해 28*28의 값을 사용하면 됨.
# (28, 28) --> 28x28 1차원 배열 데이터와 같습니다.
model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28, 512)) # fulluy Connected Layer
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(28*28, 256)) # fulluy Connected Layer
model.add_module('relu1', nn.ReLU())
model.add_module('fc3', nn.Linear(28*28, 256, 10)) # fulluy Connected Layer

model = nn.Linear(28*28, 10)
# 계산해줘 쿠다야
model = model.to(device)
# F.cross_entropy() 차이점...
loss_fn = nn.CrossEntropyLoss() # softmax(wx+b)
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

train(50)
torch.save(model,'mnist_cnn.pth')
torch.save(model.state_dict, 'mnist_cnn.pt') ##
model = torch.load('./mnist_clf.pth', weights_only=False) # cnn/dnn architecture포함

model = CNN().to(device)
model.load_state_dict(torch.load('mnist_cnn.pt')) #saves only

##########################################################################
# inference(추론) ->  보통 실행이라고 하지 않음. 이 분야에서는 학습/추론한다고 함. 정답이 없기 때문이다.


for i in range(20 ):
    img = transform(mnist_test[i][0])
    print(mnist_test[i][1])
    plt.imshow(img, cmap='gray')
    plt.show()

    # [i] 인덱스, [0]: 이미지데이터, [1]: 라벨(GT)
    x_test = mnist_test[i][0]
    print(x_test.shape, x_test.dim, type(x_test))

    # 형태 변환을 해주어야 함.
    x_test = x_test.view(-1, 784)
    prediction = model(x_test)
    print(prediction) #
    prob = F.softmax(prediction, dim=1)
    print(prob)
    print(torch.argmax(prob, dim=1)) # list에서 가장 큰 값이 있는 인덱스를 반환하는 메서드
    print()

    # 총 데이터 1000개
    # total number of data: 1000
    # 1000개 중에 900개 맞추면 0.9라서 90%
    # correct: 900 --> 900/1000 --> 0.9 --> 90%
    # 맞추는거 확인하는 변수 correct
    correct = 0
    print(len(mnist_test))
    for i in range(len(mnist_test)):
        x_test = mnist_test[i][0]
        x_test = x_test.view(-1, 784)
        prediction =model(x_test)
        prob = F.softmax(prediction, dim=1)
        gt = mnist_test[i][1]
        if torch.argmax(prob, dim=1) == gt:
            correct += 1
    print(f'Accuracy: {(correct/len(mnist_test))*100}%')