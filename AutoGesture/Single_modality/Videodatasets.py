import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, set_image_backend
from PIL import Image
import os
import math
import random
import numpy as np
# import functools
# import accimage
# set_image_backend('accimage')

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class Videodatasets(Dataset):
    def __init__(self, dataset_root, ground_truth,  typ, sample_duration=16, phase='train'):

        def get_data_list_and_label(data_df, typ):
            T = 0  # if typ == 'M' else 1
            return [(lambda arr: ('/'.join(arr[T].split('/')[1:]), int(arr[1]), int(arr[2])))(i[:-1].split(' '))
                    for i in open(data_df).readlines()]

        self.dataset_root = dataset_root
        self.sample_duration = sample_duration
        self.phase = phase

        self.transform = transforms.Compose([Normaliztion(), transforms.ToTensor()])

        lines= filter(lambda x: x[1] > 8, get_data_list_and_label(ground_truth, typ))
        self.inputs = list(lines)
    def transform_params(self, resize=(320, 240), crop_size=224, flip=0.5):
        if self.phase == 'train':
            left, top = random.randint(0, resize[0] - crop_size), random.randint(0, resize[1] - crop_size)
            is_flip = True if random.uniform(0, 1) > flip else False
        else:
            left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2
            is_flip = False
        return (left, top, left + crop_size, top + crop_size), is_flip

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        resize = (320, 240)  # default | (256, 256) may be helpful
        crop_rect, is_flip = self.transform_params(resize=resize, flip=1.0)  # no flip

        def image_to_np(image):
            """
            Returns:
                np.ndarray: Image converted to array with shape (width, height, channels)
            """
            image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
            image.copyto(image_np)
            image_np = np.transpose(image_np, (1, 2, 0))
            return image_np

        def transform(img):
            img = img.resize(resize)
            img = img.crop(crop_rect)
            if is_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return np.array(img.resize((112, 112)))
            # return image_to_np(img.resize((112, 112)))
        def Sample_Image(imgs_path, frame_count):
            sn = self.sample_duration
            if self.phase == 'train':
                f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / sn,
                                                                                       range(int(n * i / sn),
                                                                                             max(int(n * i / sn) + 1,
                                                                                                 int(n * (
                                                                                                             i + 1) / sn))))
                               for i in range(sn)]
            else:
                f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                        max(int(
                                                                                                            n * i / sn) + 1,
                                                                                                            int(n * (
                                                                                                                        i + 1) / sn))))
                               for i in range(sn)]
            sl = f(frame_count)
            frams = []
            for a in sl:
                # img = transform(accimage.Image(os.path.join(imgs_path, "%06d.jpg" % a))) #if use Accimage
                img = transform(Image.open(os.path.join(imgs_path, "%06d.jpg" % a)))
                frams.append(self.transform(img).view(3, 112, 112, 1))
            return torch.cat(frams, dim=3).type(torch.FloatTensor)
        data_path = os.path.join(self.dataset_root, self.inputs[index][0])
        clip = Sample_Image(data_path, self.inputs[index][1])
        return clip.permute(0, 3, 1, 2), self.inputs[index][2]

    def __len__(self):
        return len(self.inputs)