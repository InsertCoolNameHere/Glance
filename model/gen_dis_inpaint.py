import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2
from model.layers_inpaint import PixelShuffleUpsampler,Conv2d,CompressionBlock,init_weights,DenseConnectedBlock,DiscriminatorBlock, ResnetBlock
from collections import OrderedDict
from torchvision import transforms
import pytorch_lightning as pl

class HierarchicalGenerator(pl.LightningModule):

    def __init__(self, num_init_features, bn_size, growth_rate, ps_woReLU, level_config,
                 res_factor, max_num_feature, max_scale, num_ip_channels, lr_res, num_classes, **kwargs):
        super(HierarchicalGenerator,self).__init__()
        self.num_ip_channels = num_ip_channels
        self.lr_res = lr_res
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)
        self.lin_act = nn.Linear(3, self.lr_res * self.lr_res)
        '''cuda = torch.cuda.is_available()
        if cuda:
            self.label_embedding = self.label_embedding.cuda()
            self.lin_act = self.lin_act.cuda()'''

        self.max_upscale = max_scale
        self.num_pyramids = int(log2(self.max_upscale))

        self.current_scale_idx = self.num_pyramids - 1

        self.Upsampler = PixelShuffleUpsampler
        self.upsample_args = {'woReLU': ps_woReLU}

        denseblock_params = {
            'num_layers': None,
            'num_input_features': num_init_features,
            'bn_size': bn_size,
            'growth_rate': growth_rate,
        }

        num_features = denseblock_params['num_input_features']  # 160

        # Initiate network

        # each scale has its own init_conv - V ******************************************************
        for s in range(1, self.num_pyramids + 1):
            # in_channel=3, out_channel=160
            self.add_module('init_conv_%d' % s, Conv2d(self.num_ip_channels, num_init_features, 3))
        # RIKI RIKI THIS OUTPUT OF 160 CHANNEL SHOULD MATCH THE OUTPUT OF THE EMBEDDING....PRINT ITS SIZE
        # Each denseblock forms a pyramid - 0,1,2
        for i in range(self.num_pyramids):
            block_config = level_config[i]  # LIST LIKE [8,8,8,...,8]
            pyramid_residual = OrderedDict()

            # AT THE END OF EACH DCU, WE INCLUDE A CONV(1,1) COMPRESSION LAYER
            # NO NEED FOR THIS AT LEVEL 0
            if i != 0:
                out_planes = num_init_features

                # out_planes = num_init_features = 160
                pyramid_residual['compression_%d' % i] = CompressionBlock(in_planes=num_features, out_planes=out_planes)
                num_features = out_planes

            # serial connect blocks
            # NUM OF ELEMENTS IN block_confis IS THE NUMBER OF DCUs IN THAT PYRAMID
            # CREATING 8 DCUs
            for b, num_layers in enumerate(block_config):
                # FOR EACH DENSELY_CONNECTED_UNIT ***********************************************************
                # num_layers IS ALWAYS 8
                denseblock_params['num_layers'] = num_layers
                denseblock_params['num_input_features'] = num_features  # 160

                # DENSELY_CONNECTED_BLOCK WITH CONV(1,1) INSIDE*********************************************************
                pyramid_residual['residual_denseblock_%d' %(b + 1)] = DenseConnectedBlock(
                    res_factor=res_factor,
                    **denseblock_params)

            # conv before upsampling
            # THIS IS R
            block, num_features = self.create_finalconv(num_features, max_num_feature)

            # CREATING PYRAMID
            pyramid_residual['final_conv'] = block
            self.add_module('pyramid_residual_%d' % (i + 1),
                            nn.Sequential(pyramid_residual))

            # upsample the residual by 2 before reconstruction and next level
            self.add_module(
                'pyramid_residual_%d_residual_upsampler' % (i + 1),
                self.Upsampler(2, num_features))

            # reconstruction convolutions
            reconst_branch = OrderedDict()
            out_channels = num_features
            reconst_branch['final_conv'] = Conv2d(out_channels, 3, 3)
            self.add_module('reconst_%d' % (i + 1),
                            nn.Sequential(reconst_branch))

        init_weights(self)

    # GET V BASED ON PYRAMID-ID
    def get_init_conv(self, idx):
        """choose which init_conv based on curr_scale_idx (1-based)"""
        return getattr(self, 'init_conv_%d' % idx)

    def create_finalconv(self, in_channels, max_channels=None):
        block = OrderedDict()
        if in_channels > max_channels:
            # OUR PATH
            block['final_comp'] = CompressionBlock(in_channels, max_channels)
            block['final_conv'] = Conv2d(max_channels, max_channels, (3, 3))
            out_channels = max_channels
        else:
            block['final_conv'] = Conv2d(in_channels, in_channels, (3, 3))
            out_channels = in_channels
        return nn.Sequential(block), out_channels

    # UPSCALE_FACTOR IS THE CURRENT MODEL SCALE...FROM THE DATASET
    def forward(self, x, upscale_factor=None, blend=1.0):
        if upscale_factor is None:
            upscale_factor = self.max_scale
        else:
            valid_upscale_factors = [
                2 ** (i + 1) for i in range(self.num_pyramids)
            ]
            if upscale_factor not in valid_upscale_factors:
                print("Gen: Invalid upscaling factor {}: choose one of: {}".format(upscale_factor, valid_upscale_factors))
                raise SystemExit(1)

        # GET THE V FOR THIS UPSCALE FACTOR
        # V- COMPUTATION **********************
        feats = self.get_init_conv(log2(upscale_factor))(x)

        # THIS ENSURES WE ONLY GO DOWN THE RELEVANT PART OF THE PYRAMID
        #print(">>>>>UPSCALE:" +str(upscale_factor))
        for s in range(1, int(log2(upscale_factor)) + 1):

            # PYRAMID- COMPUTATION **********************
            feats = getattr(self, 'pyramid_residual_%d' % s)(feats) + feats

            # UPSAMPLING- COMPUTATION **********************
            # NO NEED FOR UPSAMPLING IN OUR CASE
            #feats = getattr(self, 'pyramid_residual_%d_residual_upsampler' % s)(feats)

            # RECONSTRUCTION **********************
            # reconst residual image if reached desired scale /
            # use intermediate as base_img / use blend and s is one step lower than desired scale
            if 2 ** s == upscale_factor or (blend != 1.0 and 2 ** (s + 1) == upscale_factor):
                tmp = getattr(self, 'reconst_%d' % s)(feats)
                # if using blend, upsample the second last feature via bilinear upsampling
                if (blend != 1.0 and s == self.current_scale_idx):
                    #print("HERE: SCALEID:"+str(self.current_scale_idx))
                    '''base_img = nn.functional.upsample(
                        tmp,
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True)'''
                    base_img = tmp
                if 2 ** s == upscale_factor:
                    if (blend != 1.0) and s == self.current_scale_idx + 1:
                        tmp = tmp * blend + (1 - blend) * base_img
                    output = tmp
        return output

#*******************************************************************************************************************************************
# A HIERARCHICAL DISCRIMINATOR TO MATCH THE GENERATOR
class HierarchicalDiscriminator(pl.LightningModule):

    def __init__(self, num_op_v, level_config, max_scale, num_ip_channels, lr_res, num_classes):
        super(HierarchicalDiscriminator, self).__init__()
        self.num_ip_channels = num_ip_channels
        self.lr_res = lr_res
        self.num_classes = num_classes

        '''self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)
        self.lin_act = nn.Linear(3, self.lr_res * self.lr_res)'''

        self.max_upscale = max_scale
        self.num_pyramids = int(log2(self.max_upscale))

        self.current_scale_idx = self.num_pyramids - 1

        self.level_config = level_config # LIST LIKE [64,64,128]

        # each scale has its own init_conv - V ******************************************************
        for s in range(1, self.num_pyramids + 1):
            # in_channel=3, num_op_ft=out_channel=64
            #nn.Conv2d(self.num_ip_channels, num_op_v, kernel_size=3, stride=1, padding=1)
            #nn.InstanceNorm2d
            self.add_module('init_conv_%d' % s, Conv2d(self.num_ip_channels, num_op_v, 3))

        # Each denseblock forms a pyramid - 0,1,2
        for i in range(self.num_pyramids):
            num_ip_features = level_config[i]  # LIST LIKE [64,64,128]
            pyramid_residual = DiscriminatorBlock(num_ip_features, i)

            self.add_module('pyramid_residual_%d' % (i + 1),
                            nn.Sequential(pyramid_residual))
            #print('ADDED pyramid_residual_%d' % (i+1))
            #print(pyramid_residual)


        init_weights(self)

    '''def testcuda(self):
        cuda = torch.cuda.is_available()
        print("GPU AVAILABLE:" + str(cuda))
        if cuda:
            self.input = self.input.cuda(non_blocking=True)
            self.true_hr = self.true_hr.cuda(non_blocking=True)
            self.interpolated = self.interpolated.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)
            self.net_G = self.net_G.cuda()
            self.l1_criterion = self.l1_criterion.cuda()'''

    # GET V BASED ON PYRAMID-ID
    def get_init_conv(self, idx):
        """choose which init_conv based on curr_scale_idx (1-based)"""
        return getattr(self, 'init_conv_%d' % idx)

    # UPSCALE_FACTOR IS THE CURRENT MODEL SCALE...FROM THE DATASET
    def forward(self, x, label, upscale_factor=None, blend=1.0):
        if upscale_factor is None:
            upscale_factor = self.max_upscale
        else:
            valid_upscale_factors = [
                2 ** (i + 1) for i in range(self.num_pyramids)
            ]
            if upscale_factor not in valid_upscale_factors:
                print("Invalid upscaling factor {}: choose one of: {}".format(upscale_factor, valid_upscale_factors))
                raise SystemExit(1)

        self.current_scale_idx = int(log2(upscale_factor)) - 1

        label_embedding = nn.Embedding(self.num_classes, self.num_classes).to(self.device)

        op_res = self.lr_res * upscale_factor
        lin_act = nn.Linear(3, op_res * op_res).to(self.device)

        # COMBINING OF IMAGE AND LABEL INTO A 4 CHANNEL TENSOR
        num_labels = label.size()[0]
        '''print("ISCUDA?", label.is_cuda)

        print(type(label))
        print(label)
        print(type(label_embedding))
        print(label_embedding)'''

        #RIKI ISCUDA?print("ISCUDA?", label.get_device(), self.device)
        lab_emb = label_embedding(label)
        # print("LAB", lab_emb.size())
        lab_emb = lin_act(lab_emb)
        # print("LABEL", lab_emb.size())
        lab_emb = torch.reshape(lab_emb, (num_labels, 1, op_res, op_res))

        '''if self.is_cuda:
            lab_emb = lab_emb.cuda()'''
        if self.num_ip_channels == 4:
            x = torch.cat((x, lab_emb), 1)
        # print("RIKI: gen_dis UPSCALE FACTOR: " + str(upscale_factor))
        # GET THE V FOR THIS UPSCALE FACTOR
        # V- COMPUTATION **********************
        feats = self.get_init_conv(log2(upscale_factor))(x)
        #print("AFTER V ",feats.size())
        base_img = feats
        # THIS ENSURES WE ONLY GO DOWN THE RELEVANT PART OF THE PYRAMID
        #print(">>>>>UPSCALE:" +str(upscale_factor))
        for s in range(1, int(log2(upscale_factor)) + 1):
            s1 = int(log2(upscale_factor)) - s + 1
            #print('USING pyramid_residual_%d' % s)
            # PYRAMID- COMPUTATION **********************
            feats = getattr(self, 'pyramid_residual_%d' % s1)(feats)
            #print("USING ",'pyramid_residual_%d' % s1)
            if upscale_factor > 2 and s != 1:
                base_img = getattr(self, 'pyramid_residual_%d' % s1)(base_img)
            '''if upscale_factor > 2 and s == 1:
                print("***BASE IMG IGNORES ", 'pyramid_residual_', s1)'''
            # RECONSTRUCTION **********************
            # reconst residual image if reached desired scale /
            # use intermediate as base_img / use blend and s is one step lower than desired scale

            if 2 ** s == upscale_factor:
                #tmp = feats
                #tmp = getattr(self, 'reconst_%d' % s)(feats)
                # if using blend, upsample the second last feature via bilinear upsampling
                if (blend != 1.0):
                    trans1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
                    # trans1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0, ceil_mode=True)

                    base_img = trans1(base_img)
                    #print("BLEND ON...",feats.size(), base_img.size())
                    feats = feats * blend + (1 - blend) * base_img

                output = feats

        output = torch.sigmoid(F.avg_pool2d(output, output.size()[2:])).view(output.size()[0], -1)
        return output


# SAME DISCRIMINATOR FOR ALL GENERATOR LAYERS... BAD IDEA...NOT USED ANYMORE
class SimpleDiscriminator(nn.Module):
    def __init__(self, input_shape, scale):
        super(SimpleDiscriminator, self).__init__()
        self.scale = scale
        self.input_shape = input_shape
        self.in_channels, self.in_height, self.in_width = self.input_shape

        def discriminator_block(in_filters, out_filters, first_block=False):
            #print(str(in_filters)+">>"+str(out_filters))
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            #print(str(out_filters) + ">>" + str(out_filters))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = self.in_channels #3
        for i, out_filters in enumerate([64,]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        #print(str(out_filters) + ">>>1" )
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        x=self.model(img)

        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


class InpaintGenerator(nn.Module):
    def __init__(self, num_in_channels = 4, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        out_c = 32 #64
        kernel_size = 8#4
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=num_in_channels, out_channels=out_c, kernel_size=7, padding=0),
            nn.InstanceNorm2d(out_c, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=out_c, out_channels=out_c*2, kernel_size=kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(out_c*2, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=out_c*2, out_channels=out_c*4, kernel_size=kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(out_c*4, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(out_c*4, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_c*4, out_channels=out_c*2, kernel_size=kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=out_c*2, out_channels=out_c, kernel_size=kernel_size, stride=2, padding=1),
            nn.InstanceNorm2d(out_c, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            #nn.Conv2d(in_channels=out_c, out_channels=3, kernel_size=7, padding=0),
            nn.Conv2d(in_channels=out_c, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        #print(x.size())
        x = self.middle(x)
        #print(x.size())
        x = self.decoder(x)
        #print(x.size())
        x = (torch.tanh(x) + 1) / 2
        #print(x.size())
        return x

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
