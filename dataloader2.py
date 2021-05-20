import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torchvision import transforms
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine


class MRDataset(data.Dataset):
    def __init__(self, root_dir, task, plane, train=True, transform=None, weights=None):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
        if self.train:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'train-{0}.csv'.format(task), header=None, names=['id', 'label'])
        else:
            transform = None
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

        self.transform = transform
        if weights is None:
            pos = np.sum(self.labels)
            neg = len(self.labels) - pos
            self.weights = torch.FloatTensor([1, neg / pos])
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label = self.labels[index]
        if label == 1:
            label = torch.FloatTensor([[0, 1]])
        elif label == 0:
            label = torch.FloatTensor([[1, 0]])


        #print('you are here 2')

        if self.transform:
            #array = transforms.ColorJitter(brightness=0.2)(array[0].shape)
            #print(array.shape)
            #print(array.shape)
            array = self.transform(array)
            #print(array.shape)
            #code.interact(local = locals())
            array = array.numpy()


            #print(array[0])
            #print('you are here')
            
            result = []

            for i in range(array.shape[0]):
                br = np.transpose(array[i],(1,2,0)).astype(np.uint8)
                br = transforms.ToPILImage()(br)
                br = transforms.ColorJitter(brightness=0.2)(br)
                #a.save("./temp_imgs/" + str(i)+".png",'png')
                br = np.array(br)
                #print(a.shape)
                result.append(br)
            
            result = np.stack(result)
            array = torch.FloatTensor(result)
            #print(array.shape)
            array = np.transpose(array,(0,3,2,1))

            #print(array.shape)
            #print(array.shape)
            #code.interact(local = locals()) 

        
        
        else:
            array = np.stack((array,)*3, axis=1)
            array = torch.FloatTensor(array)

        # if label.item() == 1:
        #     weight = np.array([self.weights[1]])
        #     weight = torch.FloatTensor(weight)
        # else:
        #     weight = np.array([self.weights[0]])
        #     weight = torch.FloatTensor(weight)

        return array, label, self.weights
import argparse
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args

#python dataloader.py --task "acl" --plane "sagittal" --prefix_name "temp"
if __name__ == "__main__":
    import cv2
    import code

    args = parse_arguments()

    augmentor = Compose([
        #transforms.Lambda(lambda x: np.transpose(x,(1,2,0))),
        #transforms.ToPILImage(),
        #transforms.ColorJitter(brightness=0.5),
        #transforms.Lambda(lambda x: np.array(x)),
        #transforms.Lambda(lambda x: np.resize(x,(256,256))),
        transforms.Lambda(lambda x: torch.Tensor(x)),
        RandomRotate(25),
        RandomTranslate([0.11, 0.11]),
        #RandomShear(-1,1),
        #RandomFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
        
    ])

    train_dataset = MRDataset('./data/', args.task,
                              args.plane, transform=augmentor, train=True)

    #train_dataset = MRDataset('./data/', args.task,
    #                          args.plane, transform=None, train=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=11, drop_last=False)


    #train_dataset[0][0].shape -> torch.Size([36, 3, 256, 256])
    #dataiter.next()[0].shape -> torch.Size([1, 24, 3, 256, 256])

    #dataiter = iter(train_loader)
    #data = dataiter.next()
    data = train_dataset[0]
    data_np = data[0].numpy()
    #print(data_np.shape)
    #print(data_np.shape)
    #print(data_np.shape)
    #print(data_np)
    #a = transforms.ToPILImage()(a)
    #data_np = transforms.ColorJitter(brightness=0.2)(data_np)

    for i in range(data_np.shape[0]):
        #a = np.transpose(data_np[i],(1,2,0)).astype(np.uint8)
        #a = transforms.ToPILImage()(a)
        #a = transforms.ColorJitter(brightness=0.5)(a)
        #a.save("./temp_imgs/" + str(i)+".png",'png')
        
        cv2.imwrite("./temp_imgs/" + str(i)+".png",np.transpose(data_np[i],(1,2,0)))
