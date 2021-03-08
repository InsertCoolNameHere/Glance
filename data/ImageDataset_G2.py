#LOADING THE IMAGE FILES IN A CUSTOM WAY FOR THE INPAINTING MODEL
from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections.abc import Iterable
from utils.Phase import Phase
import utils.mask_utils as Mask_Utils
from random import sample
from PIL import Image
import utils.im_manipulation as ImageManipulator

class ImageDataset_G2(Dataset):

    # READS ALL IMAGE FILE IN THE TARGET DIRECTORY AND CREATES TWO LISTS - KEYS & FILEPATHS WHOSE INDICES CORRESPOND

    def get_filenames(self, source, ext):
        img_paths = []
        for roots, dir, files in os.walk(source):
            for file in files:
                if ext in file:
                    file_abs_path = os.path.join(roots, file)

                    img_paths.append(file_abs_path)
        return img_paths


    # IMAGE UPSCALING
    # img: PIL image
    # INCREASE RESOLUTION OF IMAGE BY FACTOR OF 2^pow
    def upscaleImage(self, img, pow, method=Image.BICUBIC):
        ow, oh = img.size
        scale = 2**pow
        h = int(round(oh * scale))
        w = int(round(ow * scale))
        return img.resize((w, h), method)

    # RETURNS A HIGH RES VERSION OF THE IMAGE AT A GIVEN INDEX
    def __getitem__(self, index):
        ret_data = {}
        ret_data['scale'] = self.current_scale

        scale = self.current_scale

        # Load HIGH-RES image
        if len(self.image_fnames):
            tip = ImageManipulator.get_random_tip(self.pyramid_levels)

            filename = self.image_fnames[index]
            tokens = filename.split('/')
            ln = len(tokens)
            # print(filename, tokens)
            # QUAD HASH OF THE ENTIRE IMAGE
            image_hash = tokens[ln - 3]

            target_hash = ImageManipulator.get_candidate_tiles(image_hash, tip)

            ret_data['scale'] = self.current_scale
            ret_data['tip'] = int(tip) - self.base_level

            # FULL HIGH RESOLUTION IMAGE
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

            # THIS DEALS ONLY WITH x8 SUPER_RESOLUTION
            lr_image = ImageManipulator.downscaleImage(target_img, self.max_scale)

            ret_data['hr'] = hr_image
            ret_data['lr'] = lr_image
            ret_data['bicubic'] = ImageManipulator.downscaleImage(ret_data['lr'], 1 / scale)

            ret_data['hr'] = self.normalize_fn(ret_data['hr'])
            ret_data['lr'] = self.normalize_fn(ret_data['lr'])
            ret_data['bicubic'] = self.normalize_fn(ret_data['bicubic'])

            # THE OFFSET AT WHICH THE HR,HR' WILL HAVE TO BE CROPPED
            # WHERE TO CROP A 2x2 GRID FROM THE 8x8 GRID
            ret_data['offset'] = ImageManipulator.random_tile_offset(self.current_scale, self.mask_grid)
            # CREATING THE MASK - WHAT PERCENTAGE OF TILES ARE MISSING TRUE HR INFO?
            puncture_percent = 0.25

            # final_mask HAS percentage(eg. 20%) 0s AND THE REST 1s
            # FINAL_MASK HAS 0s IN PLACES FOR WHICH HR INFO IS MISSING
            final_mask, total_masks = Mask_Utils.create_mask_for_puncture(self.mask_grid, puncture_percent, self.tile_res)

            # SUPERIMPOSED VERSION OF HR AND HR'
            ret_data["total_masks"] = total_masks
            ret_data["mask"] = final_mask


        return ret_data

    def __len__(self):
        return len(self.image_fnames)


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

    def __init__(self, phase, albums, img_dir, img_type, mean, stddev, scales, high_res, pyramid_levels, num_inputs, mask_grid=2, ignore_files=None):

        self.phase = phase
        self.mean = mean
        self.stddev = stddev

        # THE RESOLUTION OF THE x8 ACTUAL IMAGE
        self.highest_res = high_res
        self.pyramid_levels = pyramid_levels
        self.base_level = pyramid_levels[0]
        # INITIALIZING SCALES
        self.scales = scales if isinstance(scales, Iterable) else [scales]
        #self.current_scale = self.scales[2]
        self.current_scale = self.scales[1]
        #self.current_scale = self.scales[0]
        self.current_scale_id = self.scales.index(self.current_scale)+1
        self.max_scale = np.max(self.scales)

        self.tile_res = int(high_res/self.max_scale)
        # IF PHASE IS TRAINING PHASE, WE AUGMENT THE INPUT IMAGE RANDOMLY
        #self.augmentON = self.phase == Phase.TRAIN
        self.augmentON = False
        # THE DIRECTIRY WHERE THE REAL HR IMAGES ARE
        self.img_dir = img_dir

        # COLLECTING ALL TRAINING/TESTING IMAGE PATHS
        all_images = []
        for alb in albums:
            imd = img_dir.replace("ALBUM", alb)
            all_images.extend(ImageManipulator.get_filenames(imd, img_type))
        # print("TOTAL TIFS", len(all_images))
        # IN CASE OF TEST_DATASET
        if ignore_files:
            remainder = [item for item in all_images if item not in ignore_files]
            num_sam = max(num_inputs / 30, 1)
            self.image_fnames = sample(remainder, int(num_sam))
        # IN CASE OF TRAIN DATASET
        else:
            # SELECTING A RANDOM SAMPLE OF 600 IMAGES FOR TRAINING
            self.image_fnames = sample(all_images, num_inputs)

        # ALLOWED FILE EXTENSION
        self.ext = img_type
        # Input normalization
        self.normalize_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.stddev)
        ])

        self.normalize_fn_1d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5),
                                 (0.5))
        ])

        self.mask_grid = mask_grid



