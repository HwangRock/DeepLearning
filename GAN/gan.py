import torch
import torch.nn as nn

# 생성자 네트워크 정의
class vanilla_G(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=256),  # 잠재 벡터를 256차원으로 변환
            nn.ReLU(),   # 활성화 함수로 relu를 사용한 이유는 픽셀값을 음수로 생성하는 것을 막기 위해.
            nn.Linear(in_features=256, out_features=512),  # 256차원 벡터를 512차원으로 변환
            nn.ReLU(),   # 활성화 함수로 relu를 사용한 이유는 픽셀값을 음수로 생성하는 것을 막기 위해.
            nn.Linear(in_features=512, out_features=1024),  # 512차원 벡터를 1024차원으로 변환
            nn.ReLU(),   # 활성화 함수로 relu를 사용한 이유는 픽셀값을 음수로 생성하는 것을 막기 위해.
            nn.Linear(in_features=1024, out_features=self.img_size * self.img_size),  # 1024차원 벡터를 이미지 크기로 변환
            nn.Tanh()  # 활성화 함수로 Tanh를 사용한 이유는 픽셀값을 -1~1로 정규화하기 위해서.
        )

    def forward(self, x):  # [batch, z_dim] 형태의 입력
        batch_size = x.shape[0]
        out = self.G(x)  # 생성자 네트워크를 통해 이미지 생성
        out = out.view(batch_size, 1, self.img_size, self.img_size)  # 생성된 이미지의 형태 변환
        return out

# 판별자 네트워크 정의
class vanilla_D(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.D = nn.Sequential(
            nn.Linear(in_features=self.img_size * self.img_size, out_features=1024),  # 이미지를 1024차원 벡터로 변환
            nn.LeakyReLU(negative_slope=0.2),  # 활성화 함수를 LeakyReLU를 사용하는 이유는 relu의 죽은 뉴런 때문.
            nn.Linear(in_features=1024, out_features=512),  # 1024차원 벡터를 512차원으로 변환
            nn.LeakyReLU(negative_slope=0.2), # 활성화 함수를 LeakyReLU를 사용하는 이유는 relu의 죽은 뉴런 때문.
            nn.Linear(in_features=512, out_features=256),  # 512차원 벡터를 256차원으로 변환
            nn.LeakyReLU(negative_slope=0.2),  # 활성화 함수를 LeakyReLU를 사용하는 이유는 relu의 죽은 뉴런 때문.
            nn.Linear(in_features=256, out_features=128),  # 256차원 벡터를 128차원으로 변환
            nn.LeakyReLU(negative_slope=0.2),  # 활성화 함수를 LeakyReLU를 사용하는 이유는 relu의 죽은 뉴런 때문.
            nn.Linear(in_features=128, out_features=1),  # 128차원 벡터를 이진 분류를 위해 1차원으로 변환
            nn.Sigmoid(),  # 활성화 함수로 Sigmoid를 사용한 이유는 맞을 확률과 이진분류를 위해서.
        )

    def forward(self, x):  # [batch, 1, img_size, img_size] 형태의 입력
        batch_size = x.shape[0]
        out = x.view(batch_size, -1)  # 이미지를 1차원 벡터로 변환
        out = self.D(out)  # 판별자 네트워크를 통해 진짜/가짜 여부 판별
        return out

# 생성자 손실 함수 정의
class G_Loss(nn.Module):
    def __init__(self, device):
        super(G_Loss, self).__init__()
        self.device = device
        self.criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실 함수

    def forward(self, fake):
        ones = torch.ones_like(fake).to(self.device)  # 생성된 이미지를 진짜 이미지로 판별하도록 1로 채워진 레이블 생성
        g_loss = self.criterion(fake, ones)  # 생성된 이미지가 진짜 이미지로 판별되도록 손실 계산
        return g_loss

# 판별자 손실 함수 정의
class D_Loss(nn.Module):
    def __init__(self, device):
        super(D_Loss, self).__init__()
        self.device = device
        self.criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실 함수

    def forward(self, D_real, D_fake):
        ones = torch.ones_like(D_real).to(self.device)  # 실제 이미지를 진짜 이미지로 판별하도록 1로 채워진 레이블 생성
        zeros = torch.zeros_like(D_fake).to(self.device)  # 생성된 이미지를 가짜 이미지로 판별하도록 0으로 채워진 레이블 생성

        d_real_loss = self.criterion(D_real, ones)  # 실제 이미지가 진짜 이미지로 판별되도록 손실 계산
        d_fake_loss = self.criterion(D_fake, zeros)  # 생성된 이미지가 가짜 이미지로 판별되도록 손실 계산
        d_loss = d_real_loss + d_fake_loss  # 두 손실을 합하여 판별자 전체 손실 계산

        return d_loss, d_real_loss, d_fake_loss
