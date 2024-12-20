import torch.nn as nn

#생성자 네트워크
class dcgan_G(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size

        self.G = nn.Sequential(
            # 첫 번째 레이어: 잠재 공간에서 7x7x64 특징 맵으로 변환하는 전치 컨볼루션
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=64, kernel_size=7,
                               stride=1, padding=0, bias=False),  # 출력 크기: [64, 7, 7]
            nn.BatchNorm2d(64),  # 배치 정규화
            nn.ReLU(),  # 활성화 함수로 relu를 사용한 이유는 픽셀값을 음수로 생성하는 것을 막기 위해.

            # 두 번째 레이어: 7x7x64에서 14x14x32 특징 맵으로 변환하는 전치 컨볼루션
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4,
                               stride=2, padding=1, bias=False),  # 출력 크기: [32, 14, 14]
            nn.BatchNorm2d(32),  # 배치 정규화
            nn.ReLU(),   # 활성화 함수로 relu를 사용한 이유는 픽셀값을 음수로 생성하는 것을 막기 위해.

            # 세 번째 레이어: 14x14x32에서 28x28x1 특징 맵으로 변환하는 전치 컨볼루션
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4,
                               stride=2, padding=1, bias=False),  # 출력 크기: [1, 28, 28]
            nn.Tanh()  # 활성화 함수로 Tanh를 사용한 이유는 픽셀값을 -1~1로 정규화하기 위해서.
        )

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        batch_size = x.shape[0]  # 입력 배치 크기
        x = x.view(batch_size, -1, 1, 1)
        out = self.G(x)  # 생성자 네트워크를 통과시킴
        return out  # 생성된 이미지 반환


# 판별자 네트워크
class dcgan_D(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

        self.D = nn.Sequential(
            # 첫 번째 레이어: 28x28x1 이미지를 14x14x32 특징 맵으로 변환하는 컨볼루션
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4,
                      stride=2, padding=1, bias=False),  # 출력 크기: [32, 14, 14]
            nn.BatchNorm2d(32),  # 배치 정규화
            nn.LeakyReLU(negative_slope=0.2),  # 활성화 함수를 LeakyReLU를 사용하는 이유는 relu의 죽은 뉴런 때문.

            # 두 번째 레이어: 14x14x32를 7x7x64 특징 맵으로 변환하는 컨볼루션
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                      stride=2, padding=1, bias=False),  # 출력 크기: [64, 7, 7]
            nn.BatchNorm2d(64),  # 배치 정규화
            nn.LeakyReLU(negative_slope=0.2),  # 활성화 함수를 LeakyReLU를 사용하는 이유는 relu의 죽은 뉴런 때문.

            # 세 번째 레이어: 7x7x64를 1x1x1로 변환하는 컨볼루션
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7,
                      stride=1, padding=0, bias=False),  # 출력 크기: [1, 1, 1]
            nn.Sigmoid(),  # 활성화 함수로 Sigmoid를 사용한 이유는 맞을 확률과 이진분류를 위해서.
        )

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        batch_size = x.shape[0]  # 입력 배치 크기: [batch_size, 1, 28, 28]
        out = self.D(x)  # 판별자 네트워크를 통과시킴: [batch_size, 1, 1, 1]
        out = out.squeeze()  # 출력 크기를 [batch_size, 1]로 변환
        return out  # 판별 결과 반환
