import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader


from dataset import StairDataset
from options import InferenceOptions


def inference():
    opt = InferenceOptions().parse()
    stair_dataset = StairDataset(image_dir=opt.image_dir, image_size=opt.image_size)
    # _, _, test_dataset = stair_dataset.dataset_split()
    test_dataset = stair_dataset.dataset_generate()
    
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
    loaders = test_loader
    print(f'Finish loading dataset.')

    model = torch.load(opt.pretrained_model)
    model.to(opt.device)
    criterion = opt.criterion
    model.eval()

    inference_loss = []
    softmax_pred = []
    label_true_list = []
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for data in tqdm(loaders, unit='batch'):
        img, label = data
        img, label = img.to(opt.device), label.to(opt.device)

        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label)
            label_pred = torch.argmax(output, dim=1).detach().cpu().numpy()
            label_true = label.detach().cpu().numpy()
        
        inference_loss.append(loss.item())
        softmax_pred.append(torch.softmax(output, dim=1).detach().cpu().numpy())
        label_true_list.append(label_true)
        
        true_positive += np.sum(np.logical_and(label_pred == 1, label_true == 1))
        false_positive += np.sum(np.logical_and(label_pred == 1, label_true == 0))
        true_negative += np.sum(np.logical_and(label_pred == 0, label_true == 0))
        false_negative += np.sum(np.logical_and(label_pred == 0, label_true == 1))

    avg_loss = np.mean(np.array(inference_loss))
    acc_label = {'no_stairs': true_negative / (false_positive + true_negative),
                 'stairs': true_positive / (true_positive + false_negative)}
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
    print(f'Inference Loss:{avg_loss} Accuracy:{accuracy*100:.2f}% Precision:{precision*100:.2f}% Recall:{recall*100:.2f}%')
    for label in ['no_stairs', 'stairs']:
        print(f'Label: {label:18} Recall: {acc_label[label]*100:.2f}%')
    
    # precision_list = [1]
    # recall_list = [0]    
    # for threshold in np.linspace(-100, 100, num=200):
    #     threshold = 1 / (1 + np.exp(-threshold))
    #     true_positive = 0
    #     false_positive = 0
    #     true_negative = 0
    #     false_negative = 0
    #     for softmax, label_true in zip (softmax_pred, label_true_list):
    #         label_pred = np.empty_like(label_true)
    #         for i in range(len(softmax)):
    #             label_pred[i] = 0 if softmax[i, 0] > threshold else 1

    #         true_positive += np.sum(np.logical_and(label_pred == 1, label_true == 1))
    #         false_positive += np.sum(np.logical_and(label_pred == 1, label_true == 0))
    #         true_negative += np.sum(np.logical_and(label_pred == 0, label_true == 0))
    #         false_negative += np.sum(np.logical_and(label_pred == 0, label_true == 1))
        
    #     precision = true_positive / (true_positive + false_positive)
    #     recall = true_positive / (true_positive + false_negative)
    #     precision_list.append(precision)
    #     recall_list.append(recall)
    # precision_list.append(0)
    # recall_list.append(1)
    
    # plt.figure(figsize=(10, 10), dpi=100)
    # plt.title('Precision-Recall curve')
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    # plt.xlabel('precision')
    # plt.ylabel('recall')
    # plt.plot(recall_list, precision_list)
    # plt.savefig('results/mAP.png')


if __name__ == '__main__':
    inference()
