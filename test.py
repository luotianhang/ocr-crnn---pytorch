import os
import time

import numpy as np
import torch
from PIL import Image,ImageFilter,ImageEnhance

from config import get_parser
from datalist import data_transform
from metric import RecMetric
from model import CRNN
from utils import CTCLabelConverter

'''
used for the inference
'''
from PIL import Image, ImageFilter

class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

class CarPlateRecPredict(object):
    def __init__(self):
        self.args = get_parser()
        print(f"-----------{self.args.project_name}-------------")

        use_cuda = self.args.use_cuda and torch.cuda.is_available()

        if use_cuda:
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
        else:
            torch.manual_seed(self.args.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.lines = self.output_lines('./Carplate2')

        self.model = CRNN().to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        # print(self.model)
        try:
            print("load the weight from pretrained-weight file")
            model_dict = self.model.state_dict()
            checkpoint = torch.load(self.args.pretrained_weight)
            pretrained_dict = checkpoint['model_state_dict']
            new_dict={}
            for k,v in pretrained_dict.items():
                if "repvgg" in k:
                    continue
                else:
                    new_dict[k]=v
            pretrained_dict = {k: v for k, v in new_dict.items() if
                               np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=True)
            print("Restoring the weight from pretrained-weight file \nFinished to load the weight")
        except Exception as e:
            print("can not load weight \n train the model from stratch")
            raise e

        self.convert = CTCLabelConverter(character="./data/car_plate.txt")
        self.metric = RecMetric(self.convert)

        self.predict()

    def predict(self):
        self.model.eval()
        correct = 0
        total = 0

        time_average = []

        for line in self.lines:
            target = [line.split('\\')[-1].split('.')[0]]
            img = Image.open(line).convert("RGB")

            ###########################图像模糊处理
            img=img.filter(MyGaussianBlur(radius=7))
            img.show()
            ###########################

            ###########################
            # img_bright=ImageEnhance.Brightness(img)
            # img=img_bright.enhance(10)
            # img.show()
            ###########################

            img = data_transform(img)
            img = img.unsqueeze(0)
            start = time.time()
            output = self.model(img)
            end = time.time()
            time_average.append(end - start)
            output_prob = output.permute(1, 0, 2)
            output_prob = output_prob.log_softmax(2).requires_grad_()
            result = self.metric(output_prob.permute(1, 0, 2), target)
            print(result['show_str'])
            total += len(output)
            correct += result['n_correct']
            if result['n_correct'] == 0:
                print("<-------------------------->")
            acc = correct / total

        print("final answer", acc*100,"%")
        print("total", total)
        print("correct", correct)
        print("FPS", 1/Get_Average(time_average))

    def output_lines(self, base_dir):
        output = []
        for image in os.listdir(base_dir):
            output.append(os.path.join(base_dir, image))
        return output


def Get_Average(list):
    sum = 0

    for item in list:
        sum += item

    return sum / len(list)


predict = CarPlateRecPredict()


