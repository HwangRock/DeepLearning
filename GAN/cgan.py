import torch
import torch.nn as nn

# 생성자 네트워크 정의
class conditional_G(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential(
            nn.Linear(in_features=z_dim * 2, out_features=256),  # 입력 크기는 z_dim + 조건 크기
            nn.ReLU(),   # 활성화 함수로 relu를 사용한 이유는 픽셀값을 음수로 생성하는 것을 막기 위해.
            nn.Linear(in_features=256, out_features=512),  # 256차원 벡터를 512차원으로 변환
            nn.ReLU(),   # 활성화 함수로 relu를 사용한 이유는 픽셀값을 음수로 생성하는 것을 막기 위해.
            nn.Linear(in_features=512, out_features=1024),  # 512차원 벡터를 1024차원으로 변환
            nn.ReLU(),  # 활성화 함수로 relu를 사용한 이유는 픽셀값을 음수로 생성하는 것을 막기 위해.
            nn.Linear(in_features=1024, out_features=self.img_size * self.img_size),  # 1024차원 벡터를 이미지 크기로 변환
            nn.Tanh()  # 활성화 함수로 Tanh를 사용한 이유는 픽셀값을 -1~1로 정규화하기 위해서.
        )

    def forward(self, x, c):  # x는 노이즈 벡터, c는 조건 벡터
        batch_size = x.shape[0]
        c = c.unsqueeze(1).expand(x.size())  # 조건 벡터를 노이즈 벡터와 크기를 맞춰서 확장
        x = torch.cat((x, c), dim=1)  # 노이즈와 조건을 합쳐서 생성자 입력으로 사용
        out = self.G(x)  # 생성자 네트워크를 통해 이미지 생성
        out = out.view(batch_size, 1, self.img_size, self.img_size)  # 생성된 이미지를 [batch, 1, img_size, img_size] 형태로 변환
        return out

# 판별자 네트워크 정의
class conditional_D(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.D = nn.Sequential(
            nn.Linear(in_features=self.img_size * self.img_size * 2, out_features=1024),  # 입력 크기는 이미지 크기 * 2 (이미지 + 조건)
            nn.LeakyReLU(negative_slope=0.2),   # 활성화 함수를 LeakyReLU를 사용하는 이유는 relu의 죽은 뉴런 때문.
            nn.Linear(in_features=1024, out_features=512),  # 1024차원 벡터를 512차원으로 변환
            nn.LeakyReLU(negative_slope=0.2),   # 활성화 함수를 LeakyReLU를 사용하는 이유는 relu의 죽은 뉴런 때문.
            nn.Linear(in_features=512, out_features=256),  # 512차원 벡터를 256차원으로 변환
            nn.LeakyReLU(negative_slope=0.2),   # 활성화 함수를 LeakyReLU를 사용하는 이유는 relu의 죽은 뉴런 때문.
            nn.Linear(in_features=256, out_features=128),  # 256차원 벡터를 128차원으로 변환
            nn.LeakyReLU(negative_slope=0.2),   # 활성화 함수를 LeakyReLU를 사용하는 이유는 relu의 죽은 뉴런 때문.
            nn.Linear(in_features=128, out_features=1),  # 128차원 벡터를 이진 분류를 위해 1차원으로 변환
            nn.Sigmoid()  # 활성화 함수로 Sigmoid를 사용한 이유는 맞을 확률과 이진분류를 위해서.
        )

    def forward(self, x, c):  # x는 이미지, c는 조건 벡터
        batch_size = x.shape[0]
        out = x.view(batch_size, -1)  # 이미지를 1차원 벡터로 변환
        c = c.unsqueeze(1).expand(out.size())  # 조건 벡터를 이미지 벡터와 크기를 맞춰서 확장
        out = torch.cat((out, c), dim=1)  # 이미지와 조건을 합쳐서 판별자 입력으로 사용
        out = self.D(out)  # 판별자 네트워크를 통해 진짜/가짜 여부 판별
        return out  # [batch, 1]
