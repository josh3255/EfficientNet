import torch
from efficientnet_pytorch import EfficientNet
from models import CustomModel
from loss import *
from dataset import *
import torch.optim as optim
import torch.backends.cudnn as cudnn

if __name__ == '__main__':
    model = CustomModel()

    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    else:
        print('cuda is not available')

    model.eval()

    optimizer = optim.Adam(model.parameters(), lr=3.2768e-5, weight_decay=5e-4)

    img_path = 'C:/Users/JOSH/Desktop/classification_train_data'
    label_path = 'C:/Users/JOSH/Desktop/classification_train_data/gt.txt'

    valid_img_path = 'C:/Users/JOSH/Desktop/classification_valid_data'
    valid_label_path = 'C:/Users/JOSH/Desktop/classification_valid_data/gt.txt'

    dataset = CustomDataset(img_path=img_path, label_path=label_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    valid_dataset = CustomDataset(img_path=valid_img_path, label_path=valid_label_path)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=2)

    print('data length : {}'.format(len(dataloader.dataset)))

    criterion = Loss()
    for epoch in range(100):
        total_loss = 0.0
        valid_loss = 0.0
        accuracy = 0
        for step, data in enumerate(dataloader):
            def train():
                optimizer.zero_grad()
                img, gt = data
                gt = gt.cuda()
                outputs = model(img)
                loss = criterion(outputs, gt)
                loss.backward()
                return loss
            step_loss = optimizer.step(train)
            # print('step : {} || step_loss : {}'.format(step, step_loss / 16))
            total_loss = total_loss + step_loss.item()

        for data in valid_dataloader:
            img, gt = data
            gt = gt.cuda()
            outputs = model(img)
            loss = criterion(outputs, gt)

            outputs = torch.argmax(outputs, dim=1)
            if outputs.item() == gt.item():
                accuracy = accuracy + 1

            valid_loss = valid_loss + loss

        print('epoch : {} || total_loss : {} || validation acc : {} validation_loss : {}\n'.format(epoch, total_loss, accuracy/len(valid_dataloader.dataset), valid_loss))
        # if epoch % 10 == 0:
        torch.save(model.module.state_dict(), 'D:/classifier_state_dict/effinet_' + repr(epoch) + '.pth')