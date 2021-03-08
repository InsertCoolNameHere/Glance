import os
import torch
import random
import numpy as np
import utils.im_manipulation as ImageManipulator
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from PIL import Image

this_sigma = 1.6

def create_mask(im_width, im_height, mask_width, mask_height, offset_x=None, offset_y=None):
    mask = np.zeros((im_height, im_width))
    mask_x = offset_x if offset_x is not None else random.randint(0, im_width - mask_width)
    mask_y = offset_y if offset_y is not None else random.randint(0, im_height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask

def create_mask_gray(im_width, im_height, mask_width, mask_height, offset_x=None, offset_y=None):
    mask = np.zeros((im_height, im_width))
    mask_x = offset_x if offset_x is not None else random.randint(0, im_width - mask_width)
    mask_y = offset_y if offset_y is not None else random.randint(0, im_height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask

def apply_mask_backup(self, images, edges, masks):
    images_masked = (images * (1 - masks).float()) + masks
    inputs = torch.cat((images_masked, edges), dim=1)
    outputs = self.generator(inputs)                                    # in: [rgb(3) + edge(1)]
    return outputs


def apply_mask(act_img, masks, background = None):
    """
    APPLIES A LIST OF MASKS ON A GIVEN IMAGE
    """

    # THE 1s IN THE masks REPRESENT TILES THAT HAVE HR_IMAGE, 0s FOR THE TILES FROM THE g1 IMAGE
    if background != None:
        #print("USING BACKGROUND")
        images_masked = (act_img * (masks).float()) + (background * (1-masks).float())
    else:
        images_masked = (act_img * (masks).float()) + (1-masks).float()
    return images_masked

def create_random_tile_patches(total_tiles, percentage):
    num_blank_tiles = max(int(total_tiles*percentage),1)
    tiles = random.sample(range(1, total_tiles+1), num_blank_tiles)
    return tiles


def create_mask_for_puncture(grid_size, percentage, tile_wh):
    # grid_size id the overall gxg tile area represented by the image
    #tile_wh = ImageDataset_G2.pixel_res    #WIDTH & HEIGH OF EACH TILE
    img_wh = grid_size * tile_wh #eg. 8x32

    # GET RANDOM OFFSET TO CUT OFF A 2x2 SUBSECTION OF THE WHOLE gxg TILESET
    # RANDOM TILE_NUMS
    rand_tiles = create_random_tile_patches(grid_size * grid_size, percentage)

    masks = []
    for tile1 in rand_tiles:
        tile = tile1-1
        x = int(tile / grid_size)
        y = int(tile % grid_size)

        offset_x = (x) * tile_wh
        offset_y = (y) * tile_wh
        # print(offset_x,offset_y)
        masks.append(create_mask(img_wh, img_wh, tile_wh, tile_wh, offset_x, offset_y))

    #print(len(masks))
    final_mask = np.zeros((img_wh, img_wh))
    for mask in masks:
        # print(mask.shape, target_img.size)
        final_mask += mask

    # percentage SHOULD REPRESENT TILES FOR WHICH HR IS MISSING AND HAVE TO BE LOADED FROM HR'
    # CURRENTLY, final_mask HAS percentage(eg. 20%) 1s AND THE REST 0s, SO WE NEED TO INVERT THE MASKS
    final_mask = 1 - final_mask
    #print("TILES", final_mask)
    return final_mask, len(masks)


def puncture_image(hr_img, percentage, grid_size, g1_img = None):
    """
    RETURNS NORMALIZED IMAGE
    :param hr_img: IMAGE TENSOR
    :param percentage: FRACTIONAL PERCENTAGE OF TILES TO BE REMOVED
    :param grid_size: SIZExSIZE
    """
    final_mask, total_masks = create_mask_for_puncture(grid_size, percentage)
    if total_masks == 0:
        return hr_img
    # final_mask now has 0s FOR TILES FOR WHICH TRUE HR INFORMATION IS UNAVAILABLE AND HAS TO BE LOADED FROM HR'

    # THE 1s IN THE MASK REPRESENT TILES THAT HAVE HR_IMAGE, 0s FOR THE TILES FROM THE g1 IMAGE
    hr_img = apply_mask(hr_img, F.to_tensor(final_mask).float(), g1_img)

    return hr_img, final_mask

def to_tensor(img):
    img = Image.fromarray(img,mode='F')
    img_t = F.to_tensor(img).float()
    return img_t

def get_image_edges_hr(img, mode, sigma=this_sigma):
    return get_image_edges(img, mode, this_sigma-0.8)

def get_image_edges_gen(img, mode, sigma=this_sigma):
    return get_image_edges(img, mode, this_sigma+0.5)


def get_image_edges(img, mode, sigma=this_sigma):
    """
    RETURNS A GIVEN IMAGE'S LAPLACIAN/CANNY
    :param img: IMAGE TENSOR
    :param mode: laplacian/canny
    """
    if mode == "canny":
        img_gray = rgb2gray(img)
        #img_gray = img.convert('LA')
        #return canny(img, sigma=sigma, mask=mask).astype(np.float)
        return canny(img_gray, sigma=sigma).astype(np.float)


if __name__ == '__main__':
    target_img = ImageManipulator.image_loader("/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/trai/9xjm5_20190407.tif")

    w, h = target_img.size

    # IF IMAGE PIXELS IS NOT DIVISIBLE BY SCALE, CROP THE BALANCE
    target_img = target_img.crop((0, 0, w - w % 8, h - h % 8))
    target_img,_,_ = ImageManipulator.center_crop(256, target_img)

    target_img_np = np.array(target_img)

    # Input normalization
    normalize_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ])

    target_img = normalize_fn(target_img)
    print("TARGET",target_img.size())

    img,_ = puncture_image(target_img, 0.5, 8)

    print("TARGET NP",target_img_np.shape)
    img_edges = get_image_edges(target_img_np, "canny")

    #img_edges = gray2rgb(img_edges)
    print("EDGES",img_edges.shape)

    img_edges = to_tensor(img_edges)

    print("EDGES T", img_edges.shape)
    save_image(img_edges, '/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/riki_edge.tif', normalize=False)

    #img_edges = torch.stack([img_edges, img_edges, img_edges], 0)

    #img_edges = normalize_fn(img_edges)

    print(img.size())


    # FOR DISPLAYING PUNCTURED IMAGE AND EDGES SIDE BY SIDE
    img_edges_3c = torch.cat((img_edges, img_edges, img_edges), 0)
    #print("EDGES",img_edges.shape)
    
    img_grid = torch.cat((img, img_edges_3c), 2)
    to_save = make_grid(img_grid, nrow=2, normalize=False, padding=3, pad_value=255)
    print("GRID DISPLAY", img_grid.size())
    save_image(to_save, '/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/riki_display.tif', normalize=True)


    #CREATING A 4-CHANNEL INPUT COMBINING RGB AND GRAY
    img_grid = torch.cat((img_edges,img), 0)
    print("GRID",img_grid.size())

    # nrow= NUMBER OF IMAGES IN A SINGLE ROW
    img_grid = make_grid(img_grid, nrow=1, normalize=False, padding=3, pad_value=255)

    save_image(img_grid, '/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/riki_input.tif', normalize=True)
    print("FINISHED SAVE")

