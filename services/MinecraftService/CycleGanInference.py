import sys
sys.path.append("CycleGan")
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

import imageio
import io
import torch
import numpy as np


def getCycleGanTransform(image_bytes, dataroot):
    image = imageio.imread(io.BytesIO(image_bytes))
    t_image = torch.from_numpy(np.expand_dims(np.transpose(image, (2,0,1)), 0)).cuda().float()
    model = "cycle_gan"
    opt = TestOptions()
    opt.gpu_ids = []
    opt.gpu_ids.append(0)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    opt.checkpoints_dir = "CycleGan/checkpoints"
    opt.name = dataroot
    opt.model = model
    opt.input_nc = image.shape[2]
    opt.preprocess = "none"
    opt.isTrain = False
    opt.output_nc = image.shape[2]
    opt.ngf = 64
    opt.ndf = 64
    opt.netG = 'resnet_9blocks'
    opt.netD = 'basic'
    opt.norm = 'instance'
    opt.no_dropout = 'store_true'
    opt.init_type = 'normal'
    opt.init_gain = 0.02
    opt.load_iter = 0
    opt.epoch = 'latest'
    opt.verbose = 'store_true'
    opt.suffix = ''
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    model.set_input(t_image)
    model.test()
    visuals = model.get_current_visuals()
    result = np.transpose(np.squeeze(visuals['fake_B'].data.cpu().numpy()), (1,2,0))
    return result

if __name__ == '__main__':
    with open('1.jpg', 'rb') as f:
        getTransform(f.read(), "minecraft_landscapes")
