import torch.nn as nn
# 클래스 만들기
# nn.Module로 부터 상속 받는 클래스 만들기
class BinaryClassification(nn.Module):
    # 생성자
    def __init__(self, in_dim, out_dim):
        # 부모생성자에서 함수를 가져다 쓰는
        super().__init__()
        # 객체가 던져주는 나만의 객체가 생성된다.
        self.in_dim = in_dim
        self.out_dim = out_dim
        # 멤버변수
        self.linear  =nn.Linear(in_dim, out_dim)
        self.sigmoid =nn.Sigmoid()

    """
    model = BinaryClassification(2,1)
    hx = model(x) : forward method로 수행됨.
    """
    # 메서든
    def forward(self, x):
        return self.sigmoid(self.linear(x))

    # 2학기 수업 시작
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # n: batch_size
        # 1st layer
        #input img(mnist:28x28x1)
        #conv --> (n,28,28,32) ## ## 우리가 컨볼루션을 넣을거고 ##출력을 28x28로 만들겠다
        # pool --> (n,14,14,32) ## 2x2
        # conv --> (n,14,14,64)
        # pool --> (n,7,7,64)
        self.layer1 = nn.Sequential( ##강의자료 28x28x32
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Flatten Layer (Fully Connected Layer) ## 펼치는거
        # nn.Linear(7x7x64, 10) ## (입력, 출력)
        self.fc = nn.Sequential(
                nn.Linear(in_features=7*7*64, out_features=1024, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=512, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=256, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=10, bias=True)
            )
    def forward(self,x):
        out = self.layer1(x)
        out=self.layer2(out)
        out = out.view(out.size(0), -1) #7x7x64
        out = self.fc(out)
        return out


# def CNN():
#     return None