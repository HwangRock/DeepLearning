import torch
import yaml
import sys
import os
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from fasterRCNN.utils.data_prepro import MaskDataset, collate_fn, plot_image_from_output

def main():
    print('ResNet for CIFAR10 evaluation')

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/mask_faster_rcnn.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    timestamp = "1715235364"
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))#저장된 state 불러오기.

    # 데이터 로드
    if params['task'] == "Mask":
        print('test 데이터 annotations 수 : {}'.format(len(os.listdir(params['data_files']['annotation_test_file']))))
        print('test 데이터 images 수 : {}'.format(len(os.listdir(params['data_files']['image_test_file']))))

        data_transform = transforms.Compose([  # transforms.Compose : list 내의 작업을 연달아 할 수 있게 호출하는 클래스
            transforms.ToTensor()  # ToTensor : numpy 이미지에서 torch 이미지로 변경
        ])

        test_data = MaskDataset(data_transform, params['data_files']['image_test_file'],
                                         params['data_files']['annotation_test_file'])


    test_loader = torch.utils.data.DataLoader(test_data, params['batch_size'], collate_fn=collate_fn)
    # 학습 모델 생성

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)


    # 저장된 state 불러오기
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/best.pth"))

    # TODO : 세팅값 마다 save_path를 바꾸어 로드
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    def make_prediction(model, img, threshold):
        preds = model(img) #총 이미지수
        for id in range(len(preds)):
            idx_list = []

            for idx, score in enumerate(preds[id]['scores']):
                if score > threshold:
                    idx_list.append(idx)#점수가 임계점보다 크다면 객체로 지정.

            preds[id]['boxes'] = preds[id]['boxes'][idx_list].cpu()
            preds[id]['labels'] = preds[id]['labels'][idx_list].cpu()
            preds[id]['scores'] = preds[id]['scores'][idx_list].cpu()#객체의 박스,이름,점수 저장

        return preds

    with torch.no_grad():
        for imgs, annotations in test_loader:
            imgs = list(img.to(device) for img in imgs)#이미지를 리스트로 변환해서 넣음

            preds = make_prediction(model, imgs, 0.5)#모델이 이미지를 예측한 값을 넣음.

            _idx = 1
            print("Target labels : ", annotations[_idx]['labels'])
            print("Target boxes : ", annotations[_idx]['boxes'])  # 좌상단, 우하단
            plot_image_from_output(imgs[_idx], annotations[_idx])
            print("Prediction labels : ", preds[_idx]['labels'])
            print("Prediction boxes : ", preds[_idx]['boxes'])  # 좌상단, 우하단
            plot_image_from_output(imgs[_idx], preds[_idx])

if __name__ == "__main__":
    main()