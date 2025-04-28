# -*- coding: utf-8 -*

import os
import pickle

import PIL.Image
import torch
import scipy.io as io
import numpy as np
from pathlib import Path
import shutil
import torch.utils.data
import torchvision
from torchvision.datasets import ImageFolder

__all__ = ['Cars196']

# class Cars196(ImageFolder):
#     def __getitem__(self, index):
#         img, target = super().__getitem__(index)
#         return img, target
from torchvision.transforms import transforms


class Cars196(torch.utils.data.Dataset):
    """Cars dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """

    def __init__(self, root, download=False):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._anno_filename = 'cars_annos.mat'
        self._class_txt = 'classes.txt'
        self._img_pth_split_txt = 'img_pth_train_test_split.txt'
        self._anno_filename = 'cars_annos.mat'
        self._img_filename = 'car_ims.tgz'
        self._train_folder = 'train'
        self._test_folder = 'test'

        anno_file_pth = os.path.join(self._root, self._anno_filename)
        img_file_tgz_pth = os.path.join(self._root, self._img_filename)
        img_folder_pth = os.path.join(self._root, 'car_ims')

        if self._checkIntegrity():
            print('Files already downloaded and verified.')
        else:
            if os.path.exists(anno_file_pth) and os.path.exists(img_file_tgz_pth):
                if os.path.exists(img_folder_pth):
                    self._extract_folder()
                    # self._extract()
                else:
                    raise RuntimeError(
                        'Have found dataset, but not decompression')
            else:
                if download:
                    url = None
                    self._download(url)
                else:
                    raise RuntimeError(
                        'Dataset not found. You can use download=True to download it.')

    def _checkIntegrity(self):
        """Check whether we have already processed the data.

        Returns:
            flag, bool: True if we have already processed the data.
        """
        return (
                os.path.exists(os.path.join(self._root, self._train_folder))
                and os.path.exists(os.path.join(self._root, self._test_folder)))

    def _download(self, url):
        raise NotImplementedError

    def _extract_folder(self):
        self._extract_basicinfo()
        image_path = os.path.join(self._root, 'car_ims')
        # Format of classes.txt: <class_num> <class_name>
        img_classes_name = np.genfromtxt(os.path.join(
            self._root, self._class_txt), dtype=str)
        # Format of img_pth_train_test_split.txt: <img_id> <img_pth> <img_class> <is_training_image>
        img_pth_split_info = np.genfromtxt(os.path.join(
            self._root, self._img_pth_split_txt), dtype=str)
        train_folder_pth = os.path.join(self._root, self._train_folder)
        test_folder_pth = os.path.join(self._root, self._test_folder)
        if not os.path.isdir(train_folder_pth):
            os.mkdir(train_folder_pth)
        if not os.path.isdir(test_folder_pth):
            os.mkdir(test_folder_pth)
        for each_name in img_classes_name:
            os.mkdir(os.path.join(train_folder_pth, each_name[1]))
            os.mkdir(os.path.join(test_folder_pth, each_name[1]))
        for idx, each_img in enumerate(img_pth_split_info):
            img_pth = each_img[1]
            img_class = each_img[2]
            img_is_test = each_img[3]
            img_class_name = img_classes_name[int(img_class) - 1][1]
            source = os.path.join(self._root, img_pth)
            is_train_pth = train_folder_pth
            is_test_pth = test_folder_pth

            if int(img_is_test) == 1:
                target = os.path.join(is_test_pth, img_class_name)
                shutil.copy(source, target)
            else:
                target = os.path.join(is_train_pth, img_class_name)
                shutil.copy(source, target)

    def _extract_basicinfo(self):
        anno_data = io.loadmat(os.path.join(self._root, self._anno_filename))
        labels = anno_data['annotations']
        class_names = anno_data['class_names']

        class_num = 1
        with open(os.path.join(self._root, self._class_txt), 'w') as f:
            for i in range(class_names.shape[1]):
                class_name = str(class_names[0, i][0]).replace(' ', '_')
                if '/' in class_name:
                    class_name = class_name.replace('/', '_')
                f.write(str(class_num) + ' ' + class_name + '\n')
                class_num += 1

        img_num = 1
        with open(os.path.join(self._root, self._img_pth_split_txt), 'w') as f:
            for j in range(labels.shape[1]):
                pth = str(labels[0, j][0])[2: -2]
                test = int(labels[0, j][6])
                clas = int(labels[0, j][5])
                f.write(str(img_num) + ' ' + str(pth) + ' ' + str(clas) + ' ' + str(test) + '\n')
                img_num += 1


class Cars196InstanceSample(ImageFolder):
    def __init__(self, folder, transform=None, target_transform=None, is_sample=True, k=4096):
        super().__init__(folder, transform=transform)

        self.is_sample = is_sample
        self.k = k
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 196
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.samples[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = super().__getitem__(index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_cars196_dataloaders(root, batch_size, val_batch_size, num_workers, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    data_dir = root
    # data_dir = './data/Cars196'
    train_dir = data_dir + '/train/'
    test_dir = data_dir + '/test/'
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


    trainset = torchvision.datasets.ImageFolder(train_dir, train_transform)
    assert (len(trainset) == 8144)
    devset = torchvision.datasets.ImageFolder(test_dir, val_transform)
    assert (len(devset) == 8041)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(devset, val_batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)
    num_data = len(trainset)

    return train_loader, test_loader, num_data



def get_cars196_dataloaders_sample(batch_size,  num_workers, k=4096,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    data_dir = 'data/Cars196/'
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_dir = data_dir + '/train/'
    test_dir = data_dir + '/test/'

    # CUB200toFolder(data_dir)
    #
    trainset = Cars196InstanceSample(train_dir, train_transform, is_sample=True, k=k)
    # print(len(trainset))
    assert (len(trainset) == 8144)
    devset = torchvision.datasets.ImageFolder(test_dir, val_transform)
    assert (len(devset) == 8041)
    num_data = len(trainset)
    # print(len(devset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(devset, batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)

    return train_loader, test_loader, num_data

if __name__ == '__main__':
    Cars196('data/cars196')
