from __future__ import print_function, division
import os
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataloaders.utils import encode_segmap,decode_segmap
# from DRnetwork_research.pytorch_deeplab_xception.mypath import Path
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
import pandas as pd
class IDRiDegmentation(Dataset):
    """
    PascalVoc dataset
    """
    seg_NUM_CLASSES = 5 #包括背景
    grade_num_classes = 5

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('idrid',kaggle=False),
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        #图片和标签位置
        self._image_dir = os.path.join(self._base_dir, 'Original_Images')
        self._cat_dir = os.path.join(self._base_dir, 'labels')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        # cvs文件所在
        _splits_dir = os.path.join(self._base_dir, 'labels')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            df = pd.read_csv(os.path.join(_splits_dir, splt + '.csv'))
            for idx in range(1,len(df)):
                _image = os.path.join(self._image_dir,splt,df.id_code.values[idx] + ".jpg")
                assert os.path.isfile(_image)
                _cat = os.path.join(self._cat_dir, splt,df.id_code.values[idx] + ".png")
                assert os.path.isfile(_cat)
                self.im_ids.append(df.labels.values[idx])
                self.images.append(_image)
                self.categories.append(_cat)
        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target, _grade= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample),_grade
            elif split == 'val':
                return self.transform_val(sample),_grade


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = cv2.imread(self.categories[index])
        _target = encode_segmap(_target)
        _grade= self.im_ids[index]
        return _img, _target,_grade

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.crop_image_from_gray(),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    # def __str__(self):
    #     return 'VOC2012(split=' + str(self.split) + ')'
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    idrid_train = IDRiDegmentation(args, split='train')

    dataloader = DataLoader(idrid_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, (sample,grade) in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp,dataset='idrid')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(311)
            plt.imshow(img_tmp)
            plt.subplot(312)
            plt.imshow(segmap)
            print(grade[jj])
            plt.subplot(313)
            plt.imshow(tmp)


        if ii == 1:
            break

    plt.show(block=True)
