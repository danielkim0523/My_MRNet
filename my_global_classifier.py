import os
import code
import torch
import torchvision
from dataloader import MRDataset
import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn import metrics
from torchvision import transforms
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine

device = torch.device("cuda:1")

def extract_predictions(task, plane, train=True):
    assert task in ['acl', 'meniscus', 'abnormal']
    assert plane in ['axial', 'coronal', 'sagittal']
    
    #models = os.listdir('/home/sjkim523/PRACTICE_1/models/')
    #model_name = list(filter(lambda name: task and plane, models))[0]
    
    model_path = f'/home/sjkim523/PRACTICE_1/models/model_rotate_and_shift_acl_sagittal_val_auc_0.8426_train_auc_0.8560_epoch_8.pth'
    print(model_path)

    mrnet = torch.load(model_path)
    _ = mrnet.eval()
    
    train_dataset = MRDataset('/home/sjkim523/PRACTICE_1/data/', 
                              task, 
                              plane, 
                              transform=False, 
                              train=train
                              )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)


    predictions = []
    labels = []

    #data = iter(train_loader)
    #data = data.next()


    with torch.no_grad():
        for i, (image, label, weight) in enumerate(train_loader):
            logit = mrnet(image.cuda().to(device))
            prediction = torch.sigmoid(logit)
            predictions.append(prediction[0][0].item())
            labels.append(label[0][0][0].item())

    return predictions, labels

task = 'acl'
results = {}

for plane in ['axial', 'coronal', 'sagittal']:
    predictions, labels = extract_predictions(task, plane)
    results['labels'] = labels
    results[plane] = predictions
    
#code.interact(local = locals()) 
X = np.zeros((len(predictions), 3))
X[:, 0] = results['axial']
X[:, 1] = results['coronal']
X[:, 2] = results['sagittal']

y = np.array(labels)

#logreg = linear_model.LinearRegression(solver='lbfgs')
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X, y)
print(logreg)
code.interact(local = locals()) 
task = 'acl'
results_val = {}

for plane in ['axial', 'coronal', 'sagittal']:
    predictions, labels = extract_predictions(task, plane, train=False)
    results_val['labels'] = labels
    results_val[plane] = predictions

X_val = np.zeros((len(predictions), 3))
X_val[:, 0] = results_val['axial']
X_val[:, 1] = results_val['coronal']
X_val[:, 2] = results_val['sagittal']
y_val = np.array(labels)


y_pred = logreg.predict_proba(X_val)[:, 1]

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
        return 2.0 / ((1/precision) + (1/recall))

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

AUC = metrics.roc_auc_score(y_val, y_pred)
precision = my_precision(y_val, y_pred)
recall = my_recall(y_val, y_pred)

accuracy = my_accuracy(y_val, y_pred)
f1_score = my_f1_score(y_val, y_pred)

print("AUC : {0} | precision : {1} | recall : {2} | accuracy  : {3} | f1_score : {4} ".format(
            AUC, precision, recall, accuracy, f1_score))
