import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import get_parser
from utils import CTCLabelConverter

'''
注意， 数据比那换的时候，千万不要pil 之后在np.array， 效率会变得非常慢
'''
convert = CTCLabelConverter(character="./data/car_plate.txt")

f = open(get_parser().alphabet, 'r', encoding="utf-8")

alphabet = ''.join([s.strip('\n') for s in f.readlines()])

alphabet = ' ' + alphabet


def convert_plate(target):
    plate = ''
    for i in target:
        plate += alphabet[i]
    return plate


data_transform = transforms.Compose([
    # transforms.Resize([36,140]),
    transforms.Resize([24, 94]),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class RecTextLineDataset(Dataset):
    def __init__(self, lines, type='train'):
        self.args = get_parser()
        self.alphabet = alphabet
        self.str2idx = {c: i for i, c in enumerate(self.alphabet)}
        self.labels = []
        # self.imgsize = (120, 32)
        self.imgsize = (94, 24)

        for line in lines:
            # TODO for windows
            # param = line.split('/')
            #
            # if len(param) == 2:
            #     image_path, gt_txt = param
            #     gt_txt = gt_txt.strip('\n')
            #     self.labels.append((image_path, gt_txt))
            # TODO for ubuntu
            # if len(line.split('/')[-1].split('.')[0]) == 8:
            self.labels.append((line, line.split('/')[-1].split('.')[0]))
        self.type = type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path, target = self.labels[index]
        # TODO for windows

        img = Image.open(img_path).convert("RGB")
        # new_img=Image.new("RGB",(36,141))
        # h=img.height
        # w=img.width
        #
        # ratio = h/w
        # img.resize()

        label = list()

        for c in target:
            label.append(self.str2idx[c])

        return img, label, len(target), target


def recCollate(batch):
    imgs = []
    labels = []
    lengths = []
    labels_words = []
    for _, sample in enumerate(batch):
        img, label, length, words = sample
        img = data_transform(img)
        imgs.append(img)
        labels.extend(label)
        lengths.append(length)
        labels_words.append(words)
    labels = np.array(labels).flatten().astype(np.int)

    return (torch.stack(imgs, 0),
            torch.from_numpy(labels),
            lengths,
            labels_words)
