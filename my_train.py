#python my_train.py -t acl -p sagittal --epochs=20 --prefix_name abc
import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

#from dataloader import MRDataset
from dataloader2 import MRDataset

import model

from sklearn import metrics
#from sklearn.metrics import confusion_matrix

import code
import matplotlib.pyplot as plt
import numpy as np

import torchvision
from torchvision import datasets, transforms

# code.interact(local = locals())
def train_model(model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every=100):
    _ = model.train()

    if torch.cuda.is_available():
        model.cuda()

    y_preds = []
    y_trues = []
    losses = []


    for i, (image, label, weight) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float())

        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0][1]))
        y_preds.append(probas[0][1].item())

        try:
            
            auc = metrics.roc_auc_score(y_trues, y_preds)

        except:
            
            auc = 0.5

        #writer.add_scalar('Train/Loss', loss_value,
                          #epoch * len(train_loader) + i)
        #writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg train loss {4} | train auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    #writer.add_scalar('Train/AUC_epoch', auc, epoch + i)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, epoch, num_epochs, writer, current_lr, log_every=20):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    losses = []

    for i, (image, label, weight) in enumerate(val_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction = model.forward(image.float())

        loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        

        y_trues.append(int(label[0][1]))
        y_preds.append(probas[0][1].item())

        #confusion_matrix(y_trues, y_preds)


        try:
            
            auc = metrics.roc_auc_score(y_trues, y_preds)
            

        except:
            
            auc = 0.5

        #writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        #writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss {4} | val auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    writer.add_scalar('Val/AUC_epoch', auc, epoch+i)

    val_loss = np.round(np.mean(losses), 4)
    val_auc = np.round(auc, 4)

    precision = my_precision(y_trues, y_preds)
    recall = my_recall(y_trues, y_preds)
    f1_score = my_f1_score(y_trues, y_preds)
    accuracy = my_accuracy(y_trues, y_preds)

    writer.add_scalar('Val/precision_epoch', precision, epoch+i)
    writer.add_scalar('Val/recall_epoch', recall, epoch+i)
    writer.add_scalar('Val/f1_score_epoch', f1_score, epoch+i)
    writer.add_scalar('Val/accuracy_epoch', accuracy, epoch+i)


    return precision, recall, f1_score, accuracy, val_loss, val_auc


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def imshow(img): #https://data-panic.tistory.com/10
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0)))

    print(np_img.shape)
    print((np.transpose(np_img,(1,2,0))).shape)


def run(args):
    log_root_folder = "./logs/{0}/{1}/".format(args.task, args.plane)
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    augmentor = Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        RandomRotate(25),
        RandomTranslate([0.11, 0.11]),
        #RandomFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])




    train_dataset = MRDataset('./data/', args.task,
                              args.plane, transform=augmentor, train=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)



    validation_dataset = MRDataset(
        './data/', args.task, args.plane, train=False)

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=-True, num_workers=0, drop_last=False)
    
    #dataloader = 1 36 3 256 256

    #for a in validation_loader:
    #    image = a[0] #image, label, weight

    #    plt.imshow(np.transpose(image, (1,2,0)))


    #   code.interact(local = locals())
        #image.shape
        #plt.figure
        #plt.imshow(np.transpose(image.numpy(), (4,5,3)))
        #plt.show()

        #plt.figure()
        #plt.plot(data[0].t().numpy())


        

    mrnet = model.MRNet()

    images, labels, weights = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images[0])
    grid_np = grid.numpy().astype('uint8').transpose(1,2,0)
    import cv2
    cv2.imwrite('output.png', grid_np)
    writer.add_image('images', grid, 0)
    writer.add_graph(mrnet, images)
    
    torch.onnx.export(mrnet, images, "output.onnx")

    #writer.close()

    if torch.cuda.is_available():
        mrnet = mrnet.cuda()

    optimizer = optim.Adam(mrnet.parameters(), lr=args.lr, weight_decay=0.1)

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)

    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=args.gamma)

    best_val_loss = float('inf')
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every

    t_start_training = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)

        t_start = time.time()
        
        train_loss, train_auc = train_model(
            mrnet, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every)
        precision, recall, f1_score, accuracy, val_loss, val_auc = evaluate_model(
            mrnet, validation_loader, epoch, num_epochs, writer, current_lr)

        

        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        print("val precision : {0} | val recall : {1} | val accuracy : {2} | val f1 score  : {3} | train loss : {4} | train auc {5} | val loss {6} | val auc {7} | elapsed time {8} s".format(
            precision, recall, accuracy, f1_score, train_loss, train_auc, val_loss, val_auc, delta))


        iteration_change_loss += 1
        print('-' * 30)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            if bool(args.save_model):
                file_name = f'model_{args.prefix_name}_{args.task}_{args.plane}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch+1}.pth'
                for f in os.listdir('./models/'):
                    if (args.task in f) and (args.plane in f) and (args.prefix_name in f):
                        os.remove(f'./models/{f}')
                torch.save(mrnet, f'./models/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break

    #writer.add_scalar('Val/precision_epoch', precision, epoch)
    #writer.add_scalar('Val/recall_epoch', recall, epoch)
    #writer.add_scalar('Val/recall_epoch', accuracy, epoch)
    #writer.add_scalar('Val/f1_epoch', f1_score, epoch)
    #writer.close


    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument('--prefix_name', type=str, required=True)
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=str,
                        default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args


def my_precision(y_trues, y_preds):
    true_positive = 0
    false_positive = 0
    for (i, p) in enumerate(y_preds):

        if p >= 0.5:
            p = 1
        else:
            p = 0 

        if p == 1 and y_trues[i] == 1:
            true_positive += 1
        elif p == 1 and y_trues[i] != 1:
            false_positive += 1
    try: 
        return true_positive / (true_positive + false_positive)
    
    except:
        return 0

def my_recall(y_trues, y_preds):
    true_positive = 0
    false_negative = 0 
    for (i,p) in enumerate(y_preds):

        if p >= 0.5:
            p = 1
        else:
            p = 0

        if p == 1 and y_trues[i] == 1:
            true_positive += 1 
        elif p != 1 and y_trues[i] == 1: 
            false_negative += 1 
    try: 
        return true_positive / (true_positive + false_negative)

    except:
        return 0

def my_f1_score(y_trues, y_preds):
    precision = my_precision(y_trues, y_preds)
    recall = my_recall(y_trues, y_preds)
    
    try:
        return 2.0 / (1/precision * 1/recall)

    except:
        return 0

def my_accuracy(y_trues, y_preds):
    
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    for (i,p) in enumerate(y_preds):
        
        if p >= 0.5:
            p = 1
        else:
            p = 0

        if p == 1 and y_trues[i] == 1:
            true_positive += 1 
        elif p != 1 and y_trues[i] != 1:
            true_negative += 1
        elif p == 1 and y_trues[i] != 1:
            false_positive += 1
        elif p != 1 and y_trues[i] == 1: 
            false_negative += 1

    try:
        return (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative) 
    except:
        return 0


if __name__ == "__main__":
    args = parse_arguments()
    run(args)


