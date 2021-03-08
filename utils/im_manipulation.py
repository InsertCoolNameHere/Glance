import os
import torch.nn as nn
from numpy import random
from PIL import Image
import random
import torch
import torchvision.transforms.functional as F
from torchvision.utils import save_image, make_grid
from skimage import img_as_float
from skimage.color import rgb2ycbcr
#from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.crop_utils import *
from utils.gdal_utils import *

# RETURNS ALL FILENAMES IN THE GIVEN DIRECTORY IN AN ARRAY
def get_filenames(source, ext):
    img_paths = []
    for roots, dir, files in os.walk(source):
        for file in files:
            if file.endswith(ext):
                file_abs_path = os.path.join(roots, file)
                '''tokens = file_abs_path.split('/');
                ln = len(tokens)
                geohash = tokens[ln - 2]

                f_tokens = file.split('$')
                date = f_tokens[0]'''

                img_paths.append(file_abs_path)
    #print("RETURNING...",len(img_paths))
    return img_paths

def get_valid_filenames(source, ext):
    img_paths = []
    rejects = []
    i = 0
    for roots, dir, files in os.walk(source):
        for file in files:
            i+=1
            if i%200 == 0:
                print(i)
            if file.endswith(ext):
                file_abs_path = os.path.join(roots, file)
                # FULL HIGH RESOLUTION IMAGE
                target_img = image_loader(file_abs_path)

                target_img = downscaleImage(target_img, 30)
                is_invalid = isCloudyImage(target_img)

                if not is_invalid:
                    img_paths.append(file_abs_path)

                else:
                    #print("REJECTED ",file_abs_path)
                    rejects.append(file_abs_path)
    # print("RETURNING...",len(img_paths))
    return img_paths, rejects

def isCloudyImage(img, perc=0.25):
    image_array = np.asarray(img)
    image_array = image_array.reshape(image_array.shape[0] * image_array.shape[1], 3)
    white_thresh = np.array([255, 255, 255])
    black_thresh = np.array([0, 0, 0])
    nwhite = 0
    nblack = 0
    for pixel in image_array:
        if (np.all(pixel == white_thresh)):
            nwhite += 1
        if (np.all(pixel == black_thresh)):
            nblack += 1
    if (nwhite / float(len(image_array))) > perc:
        return True
    if (nblack / float(len(image_array))) > perc:
        return True

    else:
        return False


def load_dataset(args, img_format):
    files = {'train':{},'test':{}}

    for phase in ['train','test']:
        for ft in ['source','target']:
            if args[phase].train_dataset.path[ft]:
                files[phase][ft] = get_filenames(args[phase].train_dataset.path[ft], image_format=img_format)
            else:
                files[phase][ft] = []

    return files['train'],files['test']

def is_image_file(filename, ext):
    return any(filename.lower().endswith(ext))

# RETURNING A TILE FROM $ INTERNAL GEOHASHES
def get_candidate_tiles(base_hash, length):
    internal_geohashes = get_internal_quad_hashes(base_hash, length)

    tot_tiles = len(internal_geohashes)

    if tot_tiles > 1:
        tile_selected = random.sample(range(0, tot_tiles), 1)[0]
        res_list = internal_geohashes[tile_selected]
        return res_list
    else:
        return internal_geohashes[0]

    return internal_geohashes

#RANDOMLY PICKING THE PYRAMID TIP FOR THIS BATCH
def get_random_tip(numberList):
    #numberList = [0, 1, 2]
    return random.choices(numberList, weights=(1, 2, 4), k=1)[0]


def get_internal_quad_hashes(base_hash, length):
    internal_hashes = [base_hash]
    subfixes = ['0','1','2','3']
    incr = length-len(base_hash)
    tmp_list = []

    for i in range(0, incr):
        #print(i,"=========")
        for ap in internal_hashes:
            for s in subfixes:
                tmp_list.append(ap+s)
        internal_hashes = tmp_list
        #print("HERE", internal_hashes)
        tmp_list=[]
    return internal_hashes

def image_loader_gdal(filename, target_hash):
    gdal_obj = get_gdal_obj(filename)

    #print(target_hash)

    #target_hash = "02310010132"
    north, south, east, west = get_bounding_lng_lat(target_hash)


    #print(north, south, east, west)
    latlons = []
    latlons.append((west, north))
    latlons.append((east, north))
    latlons.append((east, south))
    latlons.append((west, south))
    #print(target_hash, latlons)

    pixels = get_pixel_from_lat_lon(latlons, gdal_obj)

    #print(pixels)
    cropped = crop_irregular_polygon(pixels, filename)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    #print(np.max(cropped))
    #return Image.fromarray(np.uint8(cropped)).convert(mode), target_hash
    return Image.fromarray(np.uint8(cropped))


def image_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)


def imageResize(image, pixel_res):
    new_image = image.resize((pixel_res, pixel_res))
    return new_image

# IMAGE DOWNGRADING
    # img: PIL image
    # DECREASE RESOLUTION OF IMAGE BY FACTOR OF 2^pow
def downscaleImage(img, scale, method=Image.BICUBIC):
    ow, oh = img.size

    h = int(round(oh / scale))
    w = int(round(ow / scale))
    return img.resize((w, h), method)


def random_rot90(img, r=None):
    if r is None:
        r = random.random() * 4 
    if r < 1:
        return img.transpose(Image.ROTATE_90)
    elif r < 2:
        return img.transpose(Image.ROTATE_270)
    elif r < 3:
        return img.transpose(Image.ROTATE_180)
    else:
        return img


# RANDOM FLIPPING AND ROTATION OF THE INPUT AND TARGET IMAGES
def augment_pairs(img1, img2):
    vflip = random.random() > 0.5
    hflip = random.random() > 0.5
    rot = random.random() * 4
    if hflip:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if vflip:
        img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
        img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)

    img1 = random_rot90(img1, rot)
    img2 = random_rot90(img2, rot)
    return img1, img2

# COMBINE 3 IMAGES INTO A SINGLE IMAGE-GRID
'''
def combine_images1(lr, hr, gen, scale):
    img_hr = make_grid(hr, nrow=1, normalize=False, padding=10, pad_value=0)

    img_lr = nn.functional.interpolate(lr, scale_factor=scale)

    img_lr = make_grid(img_lr, nrow=1, normalize=False, padding=10, pad_value=0)
    gen_img = make_grid(gen, nrow=1, normalize=False, padding=10, pad_value=0)

    img_grid = torch.cat((img_lr.cpu(), img_hr.cpu(), gen_img.cpu(),), -1)
    return img_grid
'''

# COMBINE 3 IMAGES INTO A SINGLE IMAGE-GRID
def combine_images(lr, hr, gen, scale):

    if scale > 0:
        img_lr = nn.functional.interpolate(lr, scale_factor=scale)
    else:
        img_lr = lr
    img_grid = torch.cat((img_lr.detach().cpu(), hr.detach().cpu(), gen.detach().cpu()), 0)
    # nrow= NUMBER OF IMAGES IN A SINGLE ROW
    img = make_grid(img_grid, nrow=3, normalize=False, padding=3, pad_value=255)

    return img

# COMBINE 3 IMAGES INTO A SINGLE IMAGE-GRID
def combine_images_2(im1, im2, im3, im4, im5, im6, n_row):

    img_grid = torch.cat((im1.detach().cpu(), im2.detach().cpu(), im3.detach().cpu(), im4.detach().cpu(), im5.detach().cpu(), im6.detach().cpu()), 0)
    # nrow= NUMBER OF IMAGES IN A SINGLE ROW
    img = make_grid(img_grid, nrow=n_row, normalize=False, padding=3, pad_value=255)

    return img

def combine_img_list(img_tuple, num_img):
    img_grid = torch.cat(img_tuple, 0)
    # nrow= NUMBER OF IMAGES IN A SINGLE ROW
    img = make_grid(img_grid, nrow=num_img, normalize=False, padding=3, pad_value=255)

    return img

# COMBINE 4 IMAGES INTO A SINGLE IMAGE-GRID...DURING TESTING
def combine_images_test(lr, hr, bic, gen, scale):

    img_lr = nn.functional.interpolate(lr, scale_factor=scale)
    img_grid = torch.cat((img_lr.detach().cpu(), hr.detach().cpu(), bic.detach().cpu(), gen.detach().cpu()), 0)
    # nrow= NUMBER OF IMAGES IN A SINGLE ROW
    img = make_grid(img_grid, nrow=4, normalize=False, padding=3, pad_value=255)

    return img

# Converts a Tensor into a Numpy array
def tensor2im(image_tensor, mean=(0.5, 0.5, 0.5), stddev=2.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy,
                                (1, 2, 0)) * stddev + np.array(mean)) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    return np.around(image_numpy).astype(np.uint8)

# Converts a Numpy array to Tensor
def im2tensor(img):
    img_t = F.to_tensor(img).float()
    return img_t

def crop_boundaries(im, cs):
    if cs > 1:
        return im[cs:-cs, cs:-cs, ...]
    else:
        return im

def mod_crop(im, scale):
    h, w = im.shape[:2]
    # return im[(h % scale):, (w % scale):, ...]
    return im[:h - (h % scale), :w - (w % scale), ...]

def eval_psnr_and_ssim_old(im1, im2, scale):
    im1_t = np.atleast_3d(img_as_float(im1))
    im2_t = np.atleast_3d(img_as_float(im2))

    if im1_t.shape[2] == 1 or im2_t.shape[2] == 1:
        im1_t = im1_t[..., 0]
        im2_t = im2_t[..., 0]

    else:
        im1_t = rgb2ycbcr(im1_t)[:, :, 0:1] / 255.0
        im2_t = rgb2ycbcr(im2_t)[:, :, 0:1] / 255.0

    if scale > 1:
        im1_t = mod_crop(im1_t, scale)
        im2_t = mod_crop(im2_t, scale)

        # NOTE conventionally, crop scale+6 pixels (EDSR, VDSR etc)
        im1_t = crop_boundaries(im1_t, int(scale) + 2)
        im2_t = crop_boundaries(im2_t, int(scale) + 2)

    psnr_val = peak_signal_noise_ratio(im1_t, im2_t)
    ssim_val = structural_similarity(
        im1_t,
        im2_t,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        data_range=1.0,
        K1=0.01,
        K2=0.03,
        sigma=1.5)

    return psnr_val, ssim_val


def eval_psnr_and_ssim(im1, im2, scale):
    im1_t = np.atleast_3d(img_as_float(im1))
    im2_t = np.atleast_3d(img_as_float(im2))

    if im1_t.shape[2] == 1 or im2_t.shape[2] == 1:
        im1_t = im1_t[..., 0]
        im2_t = im2_t[..., 0]

    else:
        im1_t = rgb2ycbcr(im1_t)[:, :, 0:1] / 255.0
        im2_t = rgb2ycbcr(im2_t)[:, :, 0:1] / 255.0



    psnr_val = peak_signal_noise_ratio(im1_t, im2_t)
    ssim_val = structural_similarity(
        im1_t,
        im2_t,
        win_size=11,
        gaussian_weights=True,
        multichannel=True,
        data_range=1.0,
        K1=0.01,
        K2=0.03,
        sigma=1.5)

    return psnr_val, ssim_val

# GIVEN AN IMAGE, CROP OUT A CENTER SQUARE SECTION FROM IT
def center_crop(crop_size, hr):
    oh_hr = ow_hr = crop_size
    w_hr, h_hr = hr.size
    offx_hr = (w_hr - crop_size) // 2
    offy_hr = (h_hr - crop_size) // 2

    return hr.crop((offx_hr, offy_hr, offx_hr + ow_hr, offy_hr + oh_hr)), offy_hr, offx_hr

# GIVEN AN IMAGE, CROP OUT A RANDOM SQUARE SECTION FROM IT
def random_crop(crop_size, hr):
    oh_hr = ow_hr = crop_size
    imw_hr, imh_hr = hr.size

    x0 = 0
    x1 = imw_hr - ow_hr + 1

    y0 = 0
    y1 = imh_hr - oh_hr + 1

    # RANDOM CROP OFFSET NW
    offy_hr = random.randint(y0, y1)
    offx_hr = random.randint(x0, x1)

    return hr.crop((offx_hr, offy_hr, offx_hr + ow_hr, offy_hr + oh_hr)), offy_hr, offx_hr

#GIVEN A GxG GRID, RANDOMLY PICK AN OFFSET FOR A CxC SUB-GRID
def random_tile_offset(grid_size, cut_size):
    #print("BOUND:",grid_size-cut_size+1)
    offy_hr = random.randint(0, grid_size-cut_size)
    offx_hr = random.randint(0, grid_size-cut_size)
    return offx_hr,offy_hr

def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()

def crop_image_tiles(hr, gen, offset, tile_wh, tile_size):
    x,y = offset
    off_x_1 = tile_wh*x
    off_x_2 = tile_wh*(x+tile_size)
    off_y_1 = tile_wh * y
    off_y_2 = tile_wh * (y + tile_size)
    return hr[:,:,off_y_1:off_y_2,off_x_1:off_x_2], gen[:,:,off_y_1:off_y_2,off_x_1:off_x_2]

def crop_image_tiles_alt(hr, gen, offset, tile_wh, tile_size):
    x,y = offset
    off_x_1 = tile_wh*x
    off_x_2 = tile_wh*(x+tile_size)
    off_y_1 = tile_wh * y
    off_y_2 = tile_wh * (y + tile_size)
    return hr[:,off_y_1:off_y_2,off_x_1:off_x_2], gen[:,off_y_1:off_y_2,off_x_1:off_x_2]

def denormalize(img):
    mean = np.asarray([0.4488, 0.4371, 0.404])
    std = np.asarray([0.0039215, 0.0039215, 0.0039215])

    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))
    im = denormalize(img[0])
    return im.unsqueeze(0)
# SINGLE IMAGE, NOT BATCH
def denormalize_img(img):
    mean = np.asarray([0.4488, 0.4371, 0.404])
    std = np.asarray([0.0039215, 0.0039215, 0.0039215])

    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    return denormalize(img)


if __name__ == "__main__":
    filename = "/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRData_Quad/co_test/02310010132_20190604.tif"  # path to raster

    ret,_ = image_loader_gdal(filename)

    ret.save("/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/crop.JPG")

if __name__ == '__main1__':
    target_img = image_loader("/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/trai/9xjm5_20190407.tif")

    w, h = target_img.size
    # IF IMAGE PIXELS IS NOT DIVISIBLE BY SCALE, CROP THE BALANCE
    target_img = target_img.crop((0, 0, w - w % 8, h - h % 8))
    normalize_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    target_img, off_y, off_x = center_crop(256, target_img)
    target_img = normalize_fn(target_img)
    target_img = target_img.unsqueeze_(0)

    hr, gen = crop_image_tiles(target_img, target_img, (6,1), 32, 2)

    save_image(target_img, '/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/tg.tif', normalize=True)
    save_image(hr, '/s/chopin/e/proj/sustain/sapmitra/super_resolution/SRImages/hr.tif', normalize=True)



