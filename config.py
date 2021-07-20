import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="ocr rec test")
    parser.add_argument('--project_name', type=str, default="CRNN 车牌文字识别")
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--pretrained_weight', type=str, default='./weights/best.pth')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--milestones', type=list, default=[10, 30, 50, 80])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_path', type=str, default="./weights/")

    parser.add_argument("--T_length", type=int, default=23)
    parser.add_argument('--alphabet', type=str, default="./data/car_plate.txt")
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)

    args = parser.parse_args()
    return args
