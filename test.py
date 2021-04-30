import torch
from dataset import *
from models import *
from collections import OrderedDict

import cv2

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

if __name__ == '__main__':
    img_path = 'C:/Users/JOSH/Desktop/classification_valid_data'
    label_path = 'C:/Users/JOSH/Desktop/classification_valid_data/gt.txt'

    dataset = CustomDataset(img_path=img_path, label_path=label_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    model = CustomModel()
    model.load_state_dict(copyStateDict(torch.load('D:/classifier_state_dict/effinet_20.pth')))
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()

    for idx, data in enumerate(dataloader):
        img, label = data

        outputs = model(img)
        outputs = torch.argmax(outputs, dim=1)
        print('model output : {}, label : {}'.format(outputs.item(), label.item()))

