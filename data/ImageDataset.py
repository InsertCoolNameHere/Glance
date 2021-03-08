#LOADING THE IMAGE FILES IN A CUSTOM WAY
from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections.abc import Iterable
from random import sample
import torch.nn as nn

from PIL import Image
import utils.im_manipulation as ImageManipulator

class ImageDataset(Dataset):

    # IMAGE UPSCALING
    # img: PIL image
    # INCREASE RESOLUTION OF IMAGE BY FACTOR OF 2^pow
    def upscaleImage(self, img, pow, method=Image.BICUBIC):
        ow, oh = img.size
        scale = 2**pow
        h = int(round(oh * scale))
        w = int(round(ow * scale))
        return img.resize((w, h), method)

    # RETURNS A LOW RES AND HIGH RES VERSION OF THE IMAGE AT A GIVEN INDEX
    def __getitem__(self, index):
        #print("FETCHING...")
        ret_data = {}
        ret_data['scale'] = self.current_scale

        scale = self.current_scale

        # Load HIGH-RES image
        if len(self.image_fnames):
            # FULL HIGH RESOLUTION IMAGE
            filename = self.image_fnames[index]

            tokens = filename.split('/')
            ln = len(tokens)
            #print(filename, tokens)
            # QUAD HASH OF THE ENTIRE IMAGE
            image_hash = tokens[ln - 3]
            #print("IMAGE_HASH: ", image_hash)

            # LIST OF PYRAMID TOPS TO BE DEALT WITH
            # BASE HASHES CAN BE 11, 12 or 13
            # THE PYRAMID TIP SELECTED RANDOMLY
            tip = ImageManipulator.get_random_tip(self.pyramid_levels)

            target_hash = ImageManipulator.get_candidate_tiles(image_hash, tip)
            #ret_data['fname'] = filename
            ret_data['scale'] = self.current_scale
            ret_data['tip'] = int(tip) - self.base_level
            target_img = ImageManipulator.image_loader_gdal(filename, target_hash)

            target_img = ImageManipulator.imageResize(target_img, self.highest_res)

            # THIS IS HOW MUCH TO DE-MAGNIFY THE MAIN IMAGE
            # eg. if target is x2, target has to be divided by 4 and source has to be divided by 8
            div_factor = self.max_scale / self.current_scale

            # LOW-RES IMAGE GENERATION

            if div_factor > 1:
                hr_image = ImageManipulator.downscaleImage(target_img, div_factor)
            else:
                hr_image = target_img
            lr_image = ImageManipulator.downscaleImage(target_img, self.max_scale)

            if self.augmentON:
                # ROTATING & FLIPPING IMAGE PAIRS
                print("AUGMENTING")
                ret_data['lr'], ret_data['hr'] = ImageManipulator.augment_pairs(
                                                            lr_image, hr_image)

            else:
                ret_data['hr'] = hr_image
                ret_data['lr'] = lr_image

            # IMAGE & FILE_NAME

            #ret_data['bicubic'] = ImageManipulator.downscaleImage(ret_data['lr'],1 / scale)
            ret_data['bicubic'] = nn.functional.interpolate(self.tensorize(ret_data['lr']).unsqueeze(0), scale_factor=scale, mode="bicubic", align_corners=True)

            ret_data['hr_fname'] = self.image_fnames[index]

            #NORMALIZED AND CONVERTED TO TENSORS

            ret_data['hr'] = self.normalize_fn(ret_data['hr'])
            ret_data['lr'] = self.normalize_fn(ret_data['lr'])
            ret_data['bicubic'] = self.simp_normalize(ret_data['bicubic'][0])
            #ret_data['bicubic'] = self.normalize_fn(ret_data['bicubic'])

            #print(ret_data['bicubic'].size(), ret_data['lr'].size(), ret_data['hr'].size())

            '''save_image(ret_data["hr"],
                       '/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/riki_mlhr.tif',
                       normalize=True)
            save_image(ret_data["hr_egdes"],
                       '/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/riki_mledge.tif',
                       normalize=False)'''

        return ret_data

    def __len__(self):
        return len(self.image_fnames)

    ''''# DISPLAY AN IMAGE AT A GIVEN PATH
    def displayImage(self, image_path):
        img = Image.open(image_path)
        plt.figure()
        plt.imshow(img)
        plt.show()

    # DISPLAY AN IMAGE OBJECT
    def displayImageMem(self, img):
        plt.figure()
        plt.imshow(img)
        plt.show()'''

    def getTransforms(self):
        transform_list = []

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    # CALLED DURING SWITCHING OF GEARS
    def set_scales(self, s_ind):
        if s_ind >= 0 and s_ind< len(self.scales):
            self.current_scale_id = s_ind
            self.current_scale = self.scales[s_ind]
            print("RESET TO "+str(self.current_scale))
        else:
            print("RIKI: SCALE ERROR")

    def copy_scales(self, scal):
        if self.scales.index(scal) >= 0 :
            self.current_scale = scal
            #print("RESET TEST SCALE TO "+str(self.current_scale))
        else:
            print("RIKI: SCALE ERROR")

    # SETTING THE LIST OF INPUT FILES EXTERNALLY...DONE FOR TESTING/EVALUATION
    def set_image_filenames(self, fnames):
        self.image_fnames = fnames

    def __init__(self, phase, albums, img_dir, img_type, mean, stddev, scales, high_res, pyramid_levels, num_inputs, num_tests, ignore_files=None):
        self.phase = phase
        self.mean = mean
        self.stddev = stddev
        self.pyramid_levels = pyramid_levels
        self.base_level = pyramid_levels[0]
        # THE RESOLUTION OF THE x8 ACTUAL IMAGE
        self.highest_res = high_res

        # INITIALIZING SCALES
        self.scales = scales if isinstance(scales, Iterable) else [scales]
        self.current_scale = self.scales[0]
        #self.current_scale = self.scales[0]
        self.current_scale_id = self.scales.index(self.current_scale)+1
        self.max_scale = np.max(self.scales)

        # IF PHASE IS TRAINING PHASE, WE AUGMENT THE INPUT IMAGE RANDOMLY
        #self.augmentON = self.phase == Phase.TRAIN
        self.augmentON = False
        # THE DIRECTIRY WHERE IMAGE ARE
        self.img_dir = img_dir

        # COLLECTING ALL TRAINING/TESTING IMAGE PATHS
        all_images = []
        for alb in albums:
            imd = img_dir.replace("ALBUM",alb)
            all_images.extend(ImageManipulator.get_filenames(imd, img_type))
        print("TOTAL TIFS", len(all_images))
        if ignore_files:
            remainder = [item for item in all_images if item not in ignore_files]
            num_sam = max(num_inputs, 1)
            self.image_fnames = sample(remainder, int(num_sam))
        else:
            # SELECTING A RANDOM SAMPLE OF 600 IMAGES FOR TRAINING
            self.image_fnames = sample(all_images, num_inputs)

        # Input normalization
        self.normalize_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.stddev)
        ])

        self.simp_normalize = transforms.Compose([
            transforms.Normalize(self.mean, self.stddev)
        ])

        self.tensorize = transforms.Compose([
            transforms.ToTensor()
        ])

        self.normalize_fn_1d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5),
                                 (0.5))
        ])

