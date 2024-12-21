import torch
import time
import os
import random
import sys
import yaml
from matplotlib import pyplot as plt

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from DNN.models.mlp import MLP


def main():
    val_accuracy = []
    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/mnist_mlp.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # 랜덤 시드 세팅
    if 'random_seed' in params:
        seed = params['random_seed']
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    torch.backends.cudnn.benchmark = True

    # 데이터 로드
    if params['task'] == "MNIST":
        # 파이토치에서 제공하는 MNIST dataset
        mnist_train_val_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                                   download=True)
        #shape 및 실제 데이터 확인
        print(mnist_train_val_dataset.data.size())
        print(mnist_train_val_dataset.targets.size())
        num=6000
        plt.imshow(mnist_train_val_dataset.data[num],cmap="Greys",interpolation="nearest")
        plt.show()
        print(mnist_train_val_dataset.targets[num])

        # data 개수 확인
        print('The number of training data : ', len(mnist_train_val_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(mnist_train_val_dataset,
                                                                   [50000, 10000])  # 5만개의 데이터를 40000, 10000개로 나눈다.

        print('The number of training data : ', len(train_dataset))
        print('The number of validation data : ', len(val_dataset))

    elif params['task'] == "CIFAR10":
        pass

    # 배치 단위로 네트워크에 데이터를 넘겨주는 Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['batch_size'], shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=params['batch_size'], shuffle=False)

    # 학습 모델 생성
    model = MLP(params['dropout_rate']).to(device)  # 모델을 지정한 device로 올려줌

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['l2_reg_lambda'])  # model.parameters -> 가중치 w들을 의미

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    summary_dir = os.path.join(out_dir, "summaries")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    writer = SummaryWriter(summary_dir) # TensorBoard를 위한 초기화
     # training 시작
    start_time = time.time()
    highest_val_acc = 0
    global_steps = 0
    print('========================================')
    print("Start training...")
    for epoch in range(params['max_epochs']):
        train_loss = 0
        train_correct_cnt = 0
        train_batch_cnt = 0
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()# iteration 마다 gradient를 0으로 초기화
            outputs = model.forward(x.view(-1, 28 * 28)) # 28 * 28 이미지를 784 features로 reshape 후 forward
            loss = criterion(outputs, y)# cross entropy loss 계산
            loss.backward()# 가중치 w에 대해 loss를 미분
            optimizer.step()# 가중치들을 업데이트

            train_loss += loss
            train_batch_cnt += 1

            _, top_pred = torch.topk(outputs, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)
            train_correct_cnt += int(torch.sum(top_pred == y))  # 맞춘 개수 카운트

            batch_total = y.size(0)
            batch_correct = int(torch.sum(top_pred == y))
            batch_acc = batch_correct / batch_total

            writer.add_scalar("Batch/Loss", loss.item(), global_steps)
            writer.add_scalar("Batch/Acc", batch_acc, global_steps)

            global_steps += 1
            if (global_steps) % 100 == 0:
                print('Epoch [{}], Step [{}], Loss: {:.4f}'.format(epoch+1, global_steps, loss.item()))

        train_acc = train_correct_cnt / len(train_dataset) * 100
        train_ave_loss = train_loss / train_batch_cnt # 학습 데이터의 평균 loss
        training_time = (time.time() - start_time) / 60
        writer.add_scalar("Train/Loss", train_ave_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)
        print('========================================')
        print("epoch:", epoch + 1, "/ global_steps:", global_steps)
        print("training dataset average loss: %.3f" % train_ave_loss)
        print("training_time: %.2f minutes" % training_time)

        # validation (for early stopping)
        val_correct_cnt = 0
        val_loss = 0
        val_batch_cnt = 0
        model.eval()
        for x, y in dev_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model.forward(x.view(-1, 28 * 28))
            loss = criterion(outputs, y)
            val_loss += loss.item()
            val_batch_cnt += 1
            _, top_pred = torch.topk(outputs, k=1, dim=-1)
            top_pred = top_pred.squeeze(dim=1)
            val_correct_cnt += int(torch.sum(top_pred == y))# 맞춘 개수 카운트

        val_acc = val_correct_cnt / len(val_dataset) * 100
        val_ave_loss = val_loss / val_batch_cnt
        print("validation dataset accuracy: %.2f" % val_acc)
        writer.add_scalar("Val/Loss", val_ave_loss, epoch)
        writer.add_scalar("Val/Acc", val_acc, epoch)
        val_accuracy.append(val_acc)
        if val_acc > highest_val_acc:# validation accuracy가 경신될 때
            save_path = checkpoint_dir + '/epoch_' + str(epoch + 1) + '.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)

            save_path = checkpoint_dir + '/best.pth'
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict()},
                       save_path)  # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping
            highest_val_acc = val_acc
        epoch += 1
    plt.plot(val_accuracy, 'r', label='val_accuracy')
    plt.show()
if __name__ == '__main__':
    main()