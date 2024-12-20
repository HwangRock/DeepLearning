import os
from bs4 import BeautifulSoup
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

class MaskDataset(object):
    def __init__(self, transforms, image_dir_path, annotation_dir_path):
        '''
        path: path to train folder or test folder
        '''
        # transform module과 img path 경로를 정의
        self.transforms = transforms
        self.image_dir_path = image_dir_path
        self.annotation_dir_path = annotation_dir_path
        self.imgs = list(sorted(os.listdir(self.image_dir_path)))


    def __getitem__(self, idx): #special method
        # load images ad masks
        file_image = self.imgs[idx]
        file_label = self.imgs[idx][:-3] + 'xml'
        img_path = os.path.join(self.image_dir_path, file_image)

        label_path = os.path.join(self.annotation_dir_path, file_label)
        img = Image.open(img_path).convert("RGB")
        #Generate Label
        target = generate_target(label_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def generate_box(obj):#물체를 특정할때 x범위, y범위를 가지기 위함.
    xmin = float(obj.find('xmin').text)
    ymin = float(obj.find('ymin').text)
    xmax = float(obj.find('xmax').text)
    ymax = float(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):#마스트 착용 상태에 따라서 구분할 클래스 3개.
    if obj.find('name').text == "without_mask":
        return 1

    elif obj.find('name').text == "with_mask":
        return 2

    elif obj.find('name').text == "mask_weared_incorrect":
        return 3


def generate_target(file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all("object")

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))#다룰 박스와 이름 저장

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)#박스와 이름 모두 텐서 변환

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels#박스와 이름의 값들 저장

        return target


def plot_image_from_output(img, annotation):#객체에 대한 결과를 보여주기 위함.
    img = img.cpu().permute(1, 2, 0)

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for idx in range(len(annotation["boxes"])):#3가지의 label에 따라서 박스색깔을 구분.
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 1:  # without_mask
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                     facecolor='none')

        elif annotation['labels'][idx] == 2:  # with_mask
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='g',
                                     facecolor='none')

        else:  # mask_weared_incorrect

            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='orange',
                                     facecolor='none')

        ax.add_patch(rect)

    plt.show()

def collate_fn(batch):
    return tuple(zip(*batch))