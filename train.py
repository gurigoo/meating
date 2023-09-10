import torch
import argparse
import os
import gc
from tqdm import tqdm
from torch import optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models import resnet
from torchvision import models
from data.dataset import CowDataset,train_transform,val_transform

gc.collect()
'''
    acc & loss
'''
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(log_path, batch_size, lr, model_path=None):
    train_data = CowDataset(r'C:\Users\hyungu_lee\Downloads\축산물 품질(QC) 이미지\Training\pre_processed', transform=train_transform, mode='train', val_ratio=0.2)
    val_data = CowDataset(r'C:\Users\hyungu_lee\Downloads\축산물 품질(QC) 이미지\Training\pre_processed',transform=val_transform, mode='val', val_ratio=0.2)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet.resnet101
    #load model
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.4)

    model.to(device)
    criterion.to(device)

    writer = SummaryWriter(log_path)

    for epoch in range(0,200):
        model.train()
        for _iter, (img, label) in enumerate(tqdm(train_loader)):
            lens=len(train_loader)
            # optimizer에 저장된 미분값을 0으로 초기화
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            pred_logit = model(img)[0]

            # loss 값 계산
            loss = criterion(pred_logit, label)
            writer.add_scalar("Loss/train", loss, epoch*lens+_iter+1)
            writer.flush()
            # Backpropagation
            loss.backward()
            optimizer.step()
            print('train loss : ',loss.data)

        model.eval()
        valid_loss, valid_acc = AverageMeter(), AverageMeter()
        for img, label in tqdm(val_loader):
            lens=len(val_loader)
            img = img.to(device)
            label = label.to(device)
            with torch.no_grad():
                pred_logit = model(img)[0]

            # loss 값 계산
            loss = criterion(pred_logit, label)

            # Accuracy 계산
            pred_label = torch.argmax(pred_logit, 1)
            acc = (pred_label == label).sum().item() / len(img)
            valid_loss.update(loss.item(), len(img))
            valid_acc.update(acc, len(img))
            
        valid_loss = valid_loss.avg
        valid_acc = valid_acc.avg
        print('epoch:',epoch,'   loss:',valid_loss,'    valid_acc:',valid_acc)
        writer.add_scalar("Loss/val",valid_loss, epoch)
        writer.add_scalar("acc/val", valid_acc, epoch)
        writer.flush()
        torch.save(model.state_dict(),os.path.join(log_path,f"{epoch:05}.pt"))

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_path',  type=str, default='./log/01')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr',  type=float, default=1e-7)
    parser.add_argument('--model_path', default=None)
    args = parser.parse_args()
    print(args)
    train(args.log_path, args.batch_size, args.lr, args.model_path)