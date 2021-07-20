import os
import shutil

import numpy as np
import torch
# from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from otherutils import convert_maxpool2d_to_softpool2d
from config import get_parser
from datalist import RecTextLineDataset, recCollate
from metric import RecMetric
from model import CRNN
from utils import CTCLabelConverter

path = 'runs'
if os.path.exists(path):
    shutil.rmtree(path)

'''

这个项目的训练中对lr特别敏感
基本完成了
'''

# writer = SummaryWriter()

best_acc = 0


class CarPlateRec(object):
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

        kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {"num_workers": 0, "pin_memory": False}

        '''
        构造DataLoader
        '''
        # win G:\datasets\car_brand\car_brand
        # ubuntu \home\car_brand\
        self.lines = self.output_lines('/home/car_brand/')

        self.num_val = int(len(self.lines) * 0.1)
        self.num_train = len(self.lines) - self.num_val

        self.train_dataloader = DataLoader(RecTextLineDataset(self.lines[:self.num_train]),
                                           batch_size=self.args.train_batch_size,
                                           shuffle=True, **kwargs,
                                           collate_fn=recCollate)
        self.test_dataloader = DataLoader(RecTextLineDataset(self.lines[self.num_train:]),
                                          batch_size=self.args.test_batch_size,
                                          shuffle=False, **kwargs,
                                          collate_fn=recCollate)

        '''
        定义模型
        '''
        self.model = CRNN().to(self.device)
        convert_maxpool2d_to_softpool2d(self.model)
        print(self.model)

        # check the torch version is above 1.6.x
        # # beacause add_graph can be only used in the below version
        # if int(torch.__version__[2]) <= 6:
        #     dummy_input = torch.randn(1, 3, 32, 120)
        #     writer.add_graph(self.model, (dummy_input,))
        if self.args.resume:
            try:
                print("load the weight from pretrained-weight file")
                model_dict = self.model.state_dict()
                checkpoint = torch.load(self.args.pretrained_weight)
                pretrained_dict = checkpoint['model_state_dict']
                new_dict = {}
                for k, v in pretrained_dict.items():
                    if "repvgg" in k:
                        continue
                    else:
                        new_dict[k] = v
                pretrained_dict = {k[7:]: v for k, v in new_dict.items() if
                                   np.shape(model_dict[k[7:]]) == np.shape(v)}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict, strict=True)
                print("Restoring the weight from pretrained-weight file \nFinished to load the weight")
            except Exception as e:
                print("can not load weight \n train the model from stratch")
                raise e
        '''
        CUDA加速
        '''
        if use_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            cudnn.enabled = True
            cudnn.benchmark = True



        '''
        构造loss目标函数
        选择优化器
        学习率变化选择
        '''
        # blank 在ctc loss中默认排在第一个
        self.loss = torch.nn.CTCLoss(reduction="mean").to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.args.milestones,
                                                         gamma=0.5)  # 这里是模型训练的关键之处，调节的好训练的块

        self.convert = CTCLabelConverter(character="./data/car_plate.txt")
        self.metric = RecMetric(self.convert)

        try:
            for epoch in range(self.args.epochs):
                self.train(epoch)
                self.test(epoch)
            torch.cuda.empty_cache()
            print("finish model training")
        # this is useful in ubuntu system
        except KeyboardInterrupt:
            torch.save({
                "model_state_dict": self.model.state_dict()
            },
                'weights/temp/best.pth'
            )
            print("model saved")

        finally:
            torch.save({
                "model_state_dict": self.model.state_dict()
            },
                'weights/temp/best.pth'
            )
            print("model saved")

    def train(self, epoch):
        self.model.train()

        correct = 0
        total = 0
        average_loss = 0
        pbar = tqdm(self.train_dataloader)
        for image, target, length, labels_words in pbar:
            image, target = image.to(self.device), target
            self.optimizer.zero_grad()
            self.input_length, self.target_length = self.sparse_tuple_for_ctc(self.args.T_length, length)
            output = self.model(image.to(self.device))
            # TNC
            output_prob = output.permute(1, 0, 2)
            output_prob = output_prob.log_softmax(2).requires_grad_()

            loss = self.loss(output_prob,
                             target,
                             input_lengths=self.input_length,
                             target_lengths=self.target_length)
            loss.backward()
            self.optimizer.step()
            average_loss += loss.item()
            # NTC
            result = self.metric(output_prob.permute(1, 0, 2), labels_words)

            total += len(output)
            correct += result['n_correct']
            acc = correct / total

            pbar.set_description(f"epoch:{epoch} "
                                 f"loss:{round(average_loss / total, 6)} "
                                 f"acc: {acc} "
                                 f"lr:{self.optimizer.param_groups[0]['lr']}")

            # writer.add_scalar('train/loss', np.average(average_loss / total), epoch)
            # writer.add_scalar('train/acc', np.average(acc), epoch)

        self.scheduler.step()

    def test(self, epoch):
        self.model.eval()
        acc = 0
        correct = 0
        total = 0
        pbar = tqdm(self.test_dataloader)
        with torch.no_grad():
            for image, target, check_target, lable_words in pbar:
                output = self.model(image.to(self.device))
                output_prob = output.permute(1, 0, 2)
                output_prob = output_prob.log_softmax(2).requires_grad_()

                result = self.metric(output_prob.permute(1, 0, 2), lable_words)

                correct += result['n_correct']
                total += len(output)
                acc = correct / total

                pbar.set_description(f"epoch:{epoch}  "
                                     f"acc: {acc}")
        # writer.add_scalar('test/acc', acc, epoch)

        global best_acc

        if best_acc < acc:
            best_acc = acc
            torch.save({
                "model_state_dict": self.model.state_dict()
            },
                'weights/temp/best.pth'
            )
            print("model saved")

        with open("train_log.txt", 'a') as f:
            f.write(str(epoch) + "-->" + str(best_acc))
            f.write('\n')

    def sparse_tuple_for_ctc(self, T_length, lengths):
        input_lengths = []
        target_lengths = []

        for ch in lengths:
            input_lengths.append(T_length)
            target_lengths.append(ch)
        return tuple(input_lengths), tuple(target_lengths)

    def output_lines(self, base_dir):
        output = []
        for file in os.listdir(base_dir):
            # # if file not in ['ccpd_base']:
            # if file in ['ccpd_weather', 'ccpd_db', 'ccpd_fn', 'ccpd_tilt', 'ccpd_challenge', 'ccpd_blur']:
                # if file in ['ccpd_weather', 'ccpd_db', 'ccpd_base', 'ccpd_fn', 'ccpd_tilt', 'ccpd_challenge', 'ccpd_blur','ccpd_weather']:
            if os.path.isdir(base_dir+'/'+file):
                for image in os.listdir(base_dir + '/' + file):
                    output.append(base_dir + "/" + file + "/" + image)

        return output


train = CarPlateRec()
