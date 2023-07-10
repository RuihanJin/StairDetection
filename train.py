import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from options import TrainOptions
from dataset import StairDataset
from model import createModel
from logger import Logger


def train():
    opt = TrainOptions().parse()
    stair_dataset = StairDataset(image_dir=opt.image_dir, image_size=opt.image_size)
    train_dataset, valid_dataset, _ = stair_dataset.dataset_split()

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
    loaders = {'train': train_loader, 'valid': valid_loader}
    print(f'Finish loading dataset.')

    model = createModel(opt)
    model.to(opt.device)

    criterion = opt.criterion
    optimizer = opt.optimizer(model.parameters(), lr=opt.lr)
    
    log = Logger(opt)
    best_acc = 0

    for epoch in range(opt.epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            train_loss = []
            valid_loss = []
            acc_pred = 0

            for data in tqdm(loaders[phase], desc=f'Epoch {epoch} {phase}', unit='batch'):
                img, label = data
                img, label = img.to(opt.device), label.to(opt.device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    label_pred = model(img)
                    loss = criterion(label_pred, label)
                    if phase == 'valid':
                        valid_loss.append(loss.item())
                    if phase == 'train':
                        train_loss.append(loss.item())
                        loss.backward()
                        optimizer.step()
                    label_pred_np = torch.argmax(label_pred, dim=1).detach().cpu().numpy()
                    label_np = label.detach().cpu().numpy()
                    acc_pred += np.sum(label_pred_np == label_np)

            if phase == 'train':
                avg_loss = np.mean(np.array(train_loss))
                acc = acc_pred / len(train_dataset)
                print(f'Epoch {epoch} Train Loss: {avg_loss} Accuracy: {acc*100:.2f}%')
            if phase == 'valid':
                avg_loss = np.mean(np.array(valid_loss))
                acc = acc_pred / len(valid_dataset)
                print(f'Epoch {epoch} Valid Loss: {avg_loss} Accuracy: {acc*100:.2f}%')
                if acc > best_acc:
                    torch.save(model, opt.model_dir)
                    best_acc = acc
            
            # add loss and accuracy information to tensorboard
            log.scalarSummary(avg_loss, acc, epoch, phase)


if __name__ == '__main__':
    train()
