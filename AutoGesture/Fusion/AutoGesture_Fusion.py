import argparse
import time
import os
import numpy as np
from tqdm import tqdm
import random
# import pprint
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import torchvision

# from network import NetworkT_CDC as network
from utils.visualizer import Visualizer
from Videodatasets_Fusion import Videodatasets_Fusion
from utils.config import Config


def Module(args):
    if args.res_layer == 'AUG':
        print("load from AutoGesture Network")
        from collections import namedtuple
        # from network import AutoGesture_searched as network
        Genotype_Net = namedtuple('Genotype',
                                  'normal8 normal_concat8 normal16 normal_concat16 normal32 normal_concat32')

        PRIMITIVES = [
            'none',
            'skip_connect',
            'TCDC06_3x3x3',
            'TCDC03avg_3x3x3',
            'conv_1x3x3',
            'TCDC06_3x1x1',
            'TCDC03avg_3x1x1',
        ]

        Genotype_Con_Unshared = namedtuple('Genotype', 'Low_Connect Mid_Connect High_Connect')

        # For stride = 2 or stride = 1
        PRIMITIVES_3x1x1 = [
            'none',
            'TCDC06_3x1x1',
            'TCDC03avg_3x1x1',
            'conv_3x1x1',
        ]

        # For stride = 4
        PRIMITIVES_5x1x1 = [
            'none',
            'TCDC06_5x1x1',
            'TCDC03avg_5x1x1',
            'conv_5x1x1',
        ]


        genotype_rgb = Genotype_Net(
                normal8=[('TCDC06_3x1x1', 1), ('TCDC06_3x3x3', 0), ('skip_connect', 0), ('skip_connect', 1),
                         ('skip_connect', 0), ('skip_connect', 1), ('TCDC06_3x1x1', 2), ('skip_connect', 3)],
                normal_concat8=range(2, 6),
                normal16=[('TCDC06_3x1x1', 1), ('skip_connect', 0), ('TCDC06_3x1x1', 1), ('TCDC06_3x3x3', 2),
                          ('TCDC06_3x3x3', 3), ('TCDC06_3x1x1', 1), ('TCDC03avg_3x3x3', 2), ('TCDC06_3x3x3', 3)],
                normal_concat16=range(2, 6),
                normal32=[('TCDC03avg_3x3x3', 1), ('skip_connect', 0), ('conv_1x3x3', 1), ('conv_1x3x3', 0),
                          ('TCDC06_3x1x1', 1), ('skip_connect', 2), ('TCDC06_3x1x1', 1), ('TCDC06_3x3x3', 0)],
                normal_concat32=range(2, 6))

        genotype_depth = Genotype_Net(
                normal8=[('conv_1x3x3', 1), ('TCDC06_3x1x1', 0), ('skip_connect', 1), ('skip_connect', 0),
                         ('skip_connect', 1), ('skip_connect', 0), ('conv_1x3x3', 2), ('skip_connect', 1)],
                normal_concat8=range(2, 6),
                normal16=[('TCDC06_3x1x1', 1), ('conv_1x3x3', 0), ('TCDC06_3x3x3', 1), ('skip_connect', 2),
                          ('TCDC06_3x1x1', 3), ('conv_1x3x3', 0), ('TCDC06_3x1x1', 1), ('conv_1x3x3', 3)],
                normal_concat16=range(2, 6),
                normal32=[('TCDC03avg_3x3x3', 1), ('TCDC06_3x1x1', 0), ('TCDC03avg_3x3x3', 2), ('TCDC06_3x1x1', 1),
                          ('TCDC03avg_3x1x1', 1), ('TCDC06_3x3x3', 3), ('TCDC03avg_3x1x1', 4), ('TCDC03avg_3x3x3', 2)],
                normal_concat32=range(2, 6))
        genotype_con_unshared = Genotype_Con_Unshared(
            Low_Connect=['TCDC03avg_3x1x1', 'none', 'TCDC03avg_5x1x1', 'TCDC06_3x1x1', 'TCDC06_3x1x1', 'none', 'none',
                         'conv_3x1x1', 'none', 'conv_5x1x1', 'none', 'none', 'TCDC06_3x1x1', 'TCDC06_3x1x1',
                         'TCDC06_3x1x1',
                         'TCDC03avg_5x1x1', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1'],
            Mid_Connect=['TCDC06_3x1x1', 'conv_3x1x1', 'none', 'conv_3x1x1', 'TCDC06_3x1x1', 'TCDC06_5x1x1',
                         'TCDC06_3x1x1',
                         'conv_3x1x1', 'TCDC03avg_3x1x1', 'conv_5x1x1', 'TCDC06_3x1x1', 'TCDC03avg_3x1x1',
                         'TCDC06_3x1x1',
                         'TCDC06_3x1x1', 'TCDC06_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC03avg_3x1x1'],
            High_Connect=['TCDC06_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_5x1x1', 'TCDC03avg_3x1x1', 'TCDC03avg_3x1x1',
                          'TCDC06_5x1x1', 'conv_3x1x1', 'none', 'conv_3x1x1', 'conv_5x1x1', 'none', 'TCDC03avg_3x1x1',
                          'TCDC03avg_3x1x1', 'none', 'TCDC06_3x1x1', 'TCDC06_5x1x1', 'TCDC03avg_3x1x1',
                          'TCDC03avg_3x1x1'])
        genotype_con_shared = Genotype_Con_Unshared(
            Low_Connect=['conv_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_5x1x1', 'TCDC06_3x1x1', 'conv_3x1x1', 'none',
                         'conv_3x1x1',
                         'conv_3x1x1', 'TCDC06_3x1x1', 'conv_5x1x1', 'none', 'TCDC06_3x1x1', 'conv_3x1x1',
                         'TCDC06_3x1x1',
                         'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC06_3x1x1'],
            Mid_Connect=['conv_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_5x1x1', 'TCDC06_3x1x1', 'conv_3x1x1', 'none',
                         'conv_3x1x1',
                         'conv_3x1x1', 'TCDC06_3x1x1', 'conv_5x1x1', 'none', 'TCDC06_3x1x1', 'conv_3x1x1',
                         'TCDC06_3x1x1',
                         'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC06_3x1x1'],
            High_Connect=['conv_3x1x1', 'TCDC03avg_3x1x1', 'TCDC06_5x1x1', 'TCDC06_3x1x1', 'conv_3x1x1', 'none',
                          'conv_3x1x1',
                          'conv_3x1x1', 'TCDC06_3x1x1', 'conv_5x1x1', 'none', 'TCDC06_3x1x1', 'conv_3x1x1',
                          'TCDC06_3x1x1',
                          'TCDC03avg_3x1x1', 'TCDC03avg_5x1x1', 'conv_3x1x1', 'TCDC06_3x1x1'])
        from network import AutoGesture_searched as Aut_network
        model_RGB = Aut_network.AutoGesture_12layers(args.init_channels8, args.init_channels16, args.init_channels32,
                                         args.num_classes, args.layers, genotype_rgb)
        model_Depth = Aut_network.AutoGesture_12layers(args.init_channels8, args.init_channels16, args.init_channels32,
                                                     args.num_classes, args.layers, genotype_depth)
        from RGBD_searched.models.AutoGesture_RGBD_searched_12layers_DiffChannels import AutoGesture_RGBD_12layers as Auto_RGBD_Diff

        model_RGBD = Auto_RGBD_Diff(args.init_channels8, args.init_channels16, args.init_channels32, args.num_classes,
                               args.layers, genotype_rgb, genotype_depth,
                               genotype_con_unshared)
        model_RGBF = Auto_RGBD_Diff(args.init_channels8, args.init_channels16, args.init_channels32, args.num_classes,
                               args.layers, genotype_rgb, genotype_depth,
                               genotype_con_unshared)
        model_FLOWD = Auto_RGBD_Diff(args.init_channels8, args.init_channels16, args.init_channels32, args.num_classes,
                                    args.layers, genotype_rgb, genotype_depth,
                                    genotype_con_unshared)

    else:
        raise Exception('Error')

    params_rgb = torch.load(args.resume_rgb)
    try:
        model_RGB.load_state_dict(params_rgb)
        print('Load RGB state_dict...')
    except:
        print('Load RGB state_dict...')
        new_state_dict = OrderedDict()
        for k, v in params_rgb.items():
            name = k[7:]
            new_state_dict[name] = v
        model_RGB.load_state_dict(new_state_dict)
    params_depth = torch.load(args.resume_depth)
    try:
        model_Depth.load_state_dict(params_depth)
        print('Load Depth state_dict...')
    except:
        print('Load Depth state_dict...')
        new_state_dict = OrderedDict()
        for k, v in params_depth.items():
            name = k[7:]
            new_state_dict[name] = v
        model_Depth.load_state_dict(new_state_dict)

    params_rgbd = torch.load(args.resume_rgbd)
    try:
        # model_RGBD.classifier = torch.nn.Linear(3072, 25) # NV
        model_RGBD.load_state_dict(params_rgbd)
        print('Load RGBD state_dict...')
    except:
        print('Load RGBD state_dict...')
        new_state_dict = OrderedDict()
        for k, v in params_rgbd.items():
            name = k[7:]
            new_state_dict[name] = v
        model_RGBD.load_state_dict(new_state_dict)

    params_rgbf = torch.load(args.resume_rgbf)
    try:
        model_RGBF.load_state_dict(params_rgbf)
        print('Load RGBF state_dict...')
    except:
        print('Load RGBF state_dict...')
        new_state_dict = OrderedDict()
        for k, v in params_rgbf.items():
            name = k[7:]
            new_state_dict[name] = v
        model_RGBF.load_state_dict(new_state_dict)

    params_flowd = torch.load(args.resume_flowd)
    try:
        model_FLOWD.load_state_dict(params_flowd)
        print('Load FLOWD state_dict...')
    except:
        print('Load FLOWD state_dict...')
        new_state_dict = OrderedDict()
        for k, v in params_flowd.items():
            name = k[7:]
            new_state_dict[name] = v
        model_FLOWD.load_state_dict(new_state_dict)

    print('Load module Finished')
    print('='*20)
    return model_RGB.cuda(), model_Depth.cuda(), model_RGBD.cuda(), model_RGBF.cuda(), model_FLOWD.cuda()

def GetData(args):
    print('Start load Data...')
    modality1 = 'rgb'
    modality2 = 'depth'
    modality3 = 'flow'
    valid_dataset = Videodatasets_Fusion(args.data_dir_root,
                                args.dataset_splits + '/{0}_valid_lst.txt'.format(modality1), modality1,
                                args.dataset_splits + '/{0}_valid_lst.txt'.format(modality2), modality2,
                                args.dataset_splits + '/{0}_valid_lst.txt'.format(modality3), modality3,
                                args.sample_duration, phase='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.testing_batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True)

    return valid_dataloader

class val_test:
    def __init__(self, args, model_RGB, model_Depth,model_RGBD, model_RGBF, model_FLOWD, val_loader):
        self.model_RGB, self.model_Depth, self.model_RGBD, self.model_RGBF, self.model_FLOWD = model_RGB, model_Depth, model_RGBD, model_RGBF, model_FLOWD
        self.args = args

        valid_acc = self.valid(1, dataloader=val_loader)
        print('valid_acc:{}'.format(valid_acc))


    def valid(self, epoch, dataloader):
        print('Validation...')
        self.model_RGB.eval()
        self.model_Depth.eval()
        self.model_RGBD.eval()
        self.model_RGBF.eval()
        self.model_FLOWD.eval()

        with torch.no_grad():
            correct = 0
            max_num = 0
            for i, (r, l, d, f) in tqdm(enumerate(dataloader)):
                outputs_r = self.model_RGB(r.cuda())
                outputs_d = self.model_Depth(d.cuda())
                outputs_rd = self.model_RGBD(r.cuda(), d.cuda())
                outputs_rf = self.model_RGBF(r.cuda(), f.cuda())
                outputs_fd = self.model_FLOWD(f.cuda(), d.cuda())


                # ---------fusion----------
                # outputs = outputs_r + outputs_d + outputs_rd
                outputs = outputs_rd + outputs_rf + outputs_fd + outputs_r + outputs_d
                # -------------------------

                pred = torch.argmax(outputs, dim=1)
                correct += (pred == l.cuda()).sum().item()
                max_num += len(d)
                acc = float(correct) / max_num
                if (i + 1) % self.args.print_freq == 0:
                    print(' Validing [%2d/%2d, %4d/%4d], Acc: %.4f \t time: @%s' % (
                        epoch, self.args.max_epochs, i, len(dataloader.dataset) / self.args.testing_batch_size, acc,
                        time.strftime('%m.%d %H:%M:%S', time.localtime(time.time()))))
        return acc

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='config.yml',
                        dest='config', help='to set the parameters')
    parser.add_argument('-r', '--resume', default='', help='load model')
    parser.add_argument('-m', '--mode', help='train or valid or test')
    parser.add_argument('-t', '--type', help='K or M')
    parser.add_argument('-g', '--gpu_ids', default="0,1", help="gpu")
    parser.add_argument('-l', '--res_layer', default=18, help="ResNet Layer")
    parser.add_argument('-i', '--init_model', default="", help="Pretrained model on 20 BN")

    parser.add_argument('--Mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='')
    parser.set_defaults(verbose=False)
    return parser.parse_args()

if __name__ == '__main__':

    seed = 123
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    args = Config(parse())
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    cudnn.benchmark = True
    # vis = Visualizer(args.visname)
    model_RGB, model_Depth, model_RGBD, model_RGBF, model_FLOWD = Module(args)

    valid_loader = GetData(args)
    val_test(args, model_RGB, model_Depth, model_RGBD, model_RGBF, model_FLOWD, valid_loader)