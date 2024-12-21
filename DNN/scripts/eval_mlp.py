import torch
import yaml
import sys
import os

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset
from DNN.models.mlp import MLP

def main():
    print('CNN for sentence classification evaluation')

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/mnist_mlp.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    timestamp = "1711601079"
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))

    # 데이터 로드
    if params['task'] == "MNIST":
        mnist_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),
                                                             download=True)
    elif params['task'] == "CIFAR10":
        pass

    # data 개수 확인
    print('The number of test data: ', len(mnist_test_dataset))

    test_loader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, batch_size=params['batch_size'], shuffle=False)

    # 학습 모델 생성
    model = MLP.to(device)  # 모델을 지정한 device로 올려줌, dropout x

    # test 시작
    model.eval()

    # 저장된 state 불러오기
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/best.pth"))

    # TODO : 세팅값 마다 save_path를 바꾸어 로드
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    correct_cnt = 0

    iter = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model.forward(x.view(-1, 28 * 28))
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)
        correct_cnt += int(torch.sum(top_pred == y))
        if iter == 0:
            total_pred = top_pred
            total_y = y
        total_pred = torch.cat((total_pred, top_pred))
        total_y = torch.cat((total_y, y))
        iter += 1

    accuracy = correct_cnt / len(mnist_test_dataset) * 100
    print("test accuracy: %.2f%%" % accuracy)

if __name__ == "__main__":
    main()