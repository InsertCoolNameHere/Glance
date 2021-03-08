
import numpy as np
import os
import os.path as osp
import random
import sys
import torch
import yaml
from utils.dotdict import DotDict
from data.ImageDataset import ImageDataset
import utils.im_manipulation as ImageManipulator
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from utils.Phase import Phase
import utils as utils
from torch.utils.data import DataLoader
from model.trainer import HierarchicalSRTrainer
from collections import defaultdict
from time import  time

configFile="../config/prosrgan.yaml" #Config File
n_cpu=8 #number of cpu threads to use during batch generation
file_separator = "/"


def testing(args):

    test_dataset = ImageDataset(phase=Phase.TEST,
                                img_dir=args.test.train_dataset.path,
                                img_type=args.xtra.img_type,
                                mean=args.test.train_dataset.mean,
                                stddev=args.train.train_dataset.stddev,
                                scales=args.data.scale,
                                high_res=args.xtra.high_res)

    testing_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )
    current_scale_indx = 0

    saved_models = [["/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/boost_exp_b/net_G_x0.pth",
               "/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/boost_exp_b/optim_G_x0.pth"],
              ["/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/boost_exp_b/net_G_x1.pth",
               "/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/boost_exp_b/optim_G_x1.pth"],
                    ["/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/boost_exp_b/net_G_x2.pth",
                     "/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/boost_exp_b/optim_G_x2.pth"]
                    ]
    output_image_path = "/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/tested_images"


    while current_scale_indx < 3:

        if len(saved_models) == current_scale_indx:
            break

        # checkpoint is the directory where checkpoints are read from and stored
        trainer = HierarchicalSRTrainer(args, testing_data_loader, test_dataset, save_dir=args.xtra.out_path)
        trainer.current_scale_id = current_scale_indx
        trainer.model_scale = args.data.scale[current_scale_indx]
        trainer.net_G.current_scale_idx = current_scale_indx

        trainer.load(saved_models[current_scale_indx])

        trainer.testcuda()

        print("RIKI: BEGIN TESTING....")

        epoch = 0

        if len(testing_data_loader)>0:
            with torch.no_grad():
                # use validation set
                trainer.set_eval()
                trainer.reset_eval_result()

                test_dataset.copy_scales(trainer.model_scale)
                print("RIKI: TESTING SCALE "+str(test_dataset.current_scale))

                for i, imgs in enumerate(testing_data_loader):
                    trainer.set_input(imgs['lr'],imgs['hr'],imgs['bicubic'],imgs['scale'])
                    #PSNR COMPUTATIONS HERE
                    test_result = trainer.evaluate_and_generate(str(i), output_image_path)

                    print(
                        'RIKI: eval at epoch %d : ' % epoch + ' | '.join([
                            '{}: {:.07f}'.format(k, v)
                            for k, v in test_result.items()
                        ]))
                    if i == 1:
                        break

        current_scale_indx += 1







if __name__ == '__main__':

    with open(configFile) as file:
        try:
            params = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(0)

    params = DotDict(params)

    if not osp.isdir(params.xtra.out_path):
        os.makedirs(params.xtra.out_path)

    #SAVING PARAMETERS FOR RESTART....NEEDS WORK
    #np.save(osp.join(params.ip.out_path, 'params'), params)

    experiment_id = osp.basename(params.xtra.out_path)

    print('experiment ID: {}'.format(experiment_id))

    #pprint(params)
    testing(params)