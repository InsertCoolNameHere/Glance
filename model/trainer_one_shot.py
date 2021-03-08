# TRAINS ONE SCALE AT A TIME AND THEN EXITS.
# EITHER STARTS FROM A BLANK MODEL - FOR x2
# OR LOADS PREVIOUS WEIGHTS AND ADDS NEW LAYERS
import math
import os
import torch
import numpy as np
from utils.im_manipulation import tensor2im, eval_psnr_and_ssim, eval_psnr_and_ssim_old
from model.gen_dis import HierarchicalGenerator, HierarchicalDiscriminator
from data.ImageDataset import ImageDataset
import utils.im_manipulation as ImageManipulator
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl
from collections import OrderedDict, defaultdict
import random
from torchvision import transforms
from utils.Phase import Phase
from torch.utils.data import DataLoader
import socket
import logging
from time import time

n_cpu = 8
file_separator = "/"


# MODULE THAT HOUSES THE GENERATOR AND DISCRIMINATOR
class HierarchicalSRTrainer(pl.LightningModule):

    # INIT OF GENERATOR AND DISCRIMINATOR
    def __init__(self, opt=None, base_dir="/nopath", save_dir="data/checkpoints", num_nodes=1, current_scale=0, isret = False):
        super(HierarchicalSRTrainer, self).__init__()

        self.input = torch.zeros(opt.train.batch_size, 3, 48, 48, dtype=torch.float32)
        self.true_hr = torch.zeros_like(self.input, dtype=torch.float32)
        self.interpolated = torch.zeros_like(self.true_hr, dtype=torch.float32)
        self.label = torch.zeros(opt.train.batch_size, 1, dtype=torch.long)
        self.ret = isret
        self.base_dir = base_dir

        tile_dir = opt.xtra.img_tile_path
        hostname = str(socket.gethostname())
        self.ip_dir = tile_dir.replace("HOSTNAME", hostname, 1)
        self.albums = opt.xtra.albums

        self.num_nodes = num_nodes
        self.opt = opt
        self.save_dir = save_dir
        std = np.asarray(self.opt.train.dataset.stddev)
        mean = np.asarray(self.opt.train.dataset.mean)
        self.denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

        # NUMBER OF BATCHES POSSIBLE IN 1 RUN OF DATASET

        self.start_epoch = self.current_epoch
        self.progress = self.start_epoch
        self.blend = 1
        # INDEX OF THE CURRENT MODEL SCALE
        self.current_scale_id = current_scale
        self.model_scale = self.opt.data.scale[self.current_scale_id]

        # Dictionary of scalewise best evaluated scores
        self.best_eval = OrderedDict([('psnr_x%d' % s, 0.0) for s in opt.data.scale])
        # Dictionary of scalewise all evaluated scores
        self.eval_dict = OrderedDict([('psnr_x%d' % s, []) for s in opt.data.scale])

        # TENSOR TO NUMPY ARRAY
        self.tensor2im = lambda t: tensor2im(t, mean=opt.train.dataset.mean, stddev=opt.train.dataset.stddev)

        opt.G.max_scale = max(opt.data.scale)

        # INITIALIZING THE GENERATOR
        self.net_G = HierarchicalGenerator(**opt.G)
        self.best_epoch = 0

        denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

        self.lr = self.opt.train.lr

        self.d_learn = self.opt.train.lr * 0.1
        self.d_decay = self.opt.train.D_lr_decay
        self.discriminator = HierarchicalDiscriminator(self.opt.Discriminator.num_op_v,
                                                       self.opt.Discriminator.level_config,
                                                       self.opt.data.scale[-1],
                                                       self.opt.G.num_ip_channels, self.opt.G.lr_res,
                                                       self.opt.G.num_classes)

        #self.l1_criterion = torch.nn.L1Loss()
        self.l1_criterion = torch.nn.MSELoss()
        #self.l1_criterion = torch.nn.SmoothL1Loss()
        self.errors_accum = defaultdict(list)

        training_dataset = ImageDataset(phase=Phase.TRAIN,
                                        albums=self.albums,
                                        img_dir=self.ip_dir,
                                        img_type=self.opt.xtra.img_type,
                                        mean=self.opt.train.dataset.mean,
                                        stddev=self.opt.train.dataset.stddev,
                                        scales=self.opt.data.scale,
                                        high_res=self.opt.xtra.high_res,
                                        pyramid_levels=self.opt.xtra.pyramid_levels,
                                        num_inputs=self.opt.xtra.num_inputs,
                                        num_tests=self.opt.xtra.num_tests,
                                        ignore_files=self.opt.ignorables)
        self.train_dataset = training_dataset

        to_ignore = []
        to_ignore.extend(training_dataset.image_fnames)
        to_ignore.extend(self.opt.ignorables)
        print("RIKI: IGNORED INVALID FILES: ", len(self.opt.ignorables))
        # LOADING TEST_DATA************************************
        self.testing_dataset = ImageDataset(phase=Phase.TEST,
                                            albums=self.albums,
                                            img_dir=self.ip_dir,
                                            img_type=self.opt.xtra.img_type,
                                            mean=self.opt.train.dataset.mean,
                                            stddev=self.opt.train.dataset.stddev,
                                            scales=self.opt.data.scale,
                                            high_res=self.opt.xtra.high_res,
                                            pyramid_levels=self.opt.xtra.pyramid_levels,
                                            num_inputs=self.opt.xtra.num_tests,
                                            num_tests=self.opt.xtra.num_tests,
                                            ignore_files=to_ignore)
        # PERFORM EVALUATION

        self.testing_data_loader = DataLoader(
            self.testing_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=n_cpu,
        )
        # END- LOADING TEST DATA********************************
        print("RIKI: TEST DATA HAS", len(self.testing_data_loader))

        self.val_dataset = ImageDataset(phase=Phase.TEST,
                                        albums=self.albums,
                                        img_dir=self.ip_dir,
                                        img_type=self.opt.xtra.img_type,
                                        mean=self.opt.train.dataset.mean,
                                        stddev=self.opt.train.dataset.stddev,
                                        scales=self.opt.data.scale,
                                        high_res=self.opt.xtra.high_res,
                                        pyramid_levels=self.opt.xtra.pyramid_levels,
                                        num_inputs=self.opt.xtra.num_vals,
                                        num_tests=self.opt.xtra.num_vals,
                                        ignore_files=to_ignore)

        if self.ret:
            self.ret = False
            self.actual_increment()
            self.set_train()
            logging.info("SWITCHING GEARS...RESTARTING EPOCH " + str(self.current_epoch) + " SCALE_ID " + str(self.current_scale_id))

        self.psnrs = 0.0
        self.losses = 0.0
        self.losses_l1 = 0.0
        self.psnr_count = 0.0

        self.train_losses = 0.0
        self.train_cnt = 0.0
        self.train_l1 = 0.0
        logging.info("RIKI: BEGINNING EPOCH: "+str(self.start_epoch)+" "+str(self.current_scale_id)+" "+str(time()))
        self.start_time = time()

    def configure_optimizers(self):
        '''if self.model_scale > 2:
            print("PARAMETERSffff::::", [self.net_G.parameters()])
            print("PARAMETERS::::",[p for p in self.net_G.parameters() if p.requires_grad])'''
        self.optimizer_G = torch.optim.Adam(
            [p for p in self.net_G.parameters() if p.requires_grad],
            lr=self.opt.train.lr,
            betas=(0.9, 0.999),
            eps=1.0e-08)

        self.optimizer_D = torch.optim.Adam(
            [p for p in self.discriminator.parameters() if p.requires_grad],
            lr=self.d_learn,
            betas=(0.9, 0.999),
            eps=1.0e-08)

        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G,mode='min',factor=0.1,patience=self.opt.train.lr_schedule_patience,
                                                                      verbose=True,min_lr=self.opt.train.D_smallest_lr)

        self.scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_D,mode='min',factor=0.1,patience=self.opt.train.lr_schedule_patience,
                                                                      verbose=True,min_lr=self.opt.train.D_smallest_lr)

        optimizers = [self.optimizer_G,self.optimizer_D]

        #'scheduler': lr_scheduler,  # The LR schduler
        #'interval': 'epoch',  # The unit of the scheduler's step size
        #'frequency': 1,  # The frequency of the scheduler
        #'reduce_on_plateau': False,  # For ReduceLROnPlateau scheduler
        schedulers = [{
                 'scheduler': self.scheduler_G,
                 'monitor': 'val_loss',
                 'interval': 'epoch',
                 'frequency': 1,
                 'strict': True,
                },
                {
                'scheduler': self.scheduler_D,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            },
                ]

        return optimizers, schedulers
        #return [self.optimizer_G, self.optimizer_D], []
        '''return (
            {"optimizer": self.optimizer_G, "lr_scheduler": self.scheduler_G, "monitor": "val_loss"},
            {"optimizer": self.optimizer_D, "lr_scheduler": self.scheduler_D, "monitor": "val_loss"}
        )'''

    def val_dataloader(self):
        # LOADING VALIDATION DATA************************************
        self.val_data_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=n_cpu,
        )
        # END- LOADING VALIDATION DATA********************************

        print("RIKI: VAL DATA HAS", len(self.val_data_loader))
        return self.val_data_loader

    def validation_step(self, batch, batch_nb):
        imgs = batch

        criterion_GAN = torch.nn.MSELoss()
        current_scale = int(imgs['scale'][0].item())

        self.set_input(imgs['lr'], imgs['hr'], imgs['bicubic'], imgs['tip'])
        # GENERATOR MAKING HR IMAGE

        bic_image = imgs['bicubic']
        self.calc_output()
        gen_hr = self.forward()
        current_blend = self.blend
        discrim_verdict = self.discriminator(gen_hr - bic_image, imgs['tip'], current_scale, current_blend)
        valid = torch.ones(discrim_verdict.size(), dtype=torch.float32).to(self.device)
        loss_GAN = criterion_GAN(discrim_verdict, valid)


        self.compute_loss(loss_GAN)

        im1 = self.tensor2im(self.true_hr)
        im2 = self.tensor2im(self.output)

        psnrval = eval_psnr_and_ssim(im1, im2, self.model_scale)[0]

        if not math.isfinite(psnrval):
            #print("RIKI: ENCOUNTERED INF...IGNORE IT")
            r=1
        else:
            self.psnrs += psnrval
            self.losses += self.loss_G
            self.losses_l1 += self.l1_loss
            self.psnr_count += 1

        #self.log("val_loss", self.loss_G, on_epoch=True, logger=True)
        self.log("val_loss", 1/psnrval, on_epoch=True, logger=True)

    def train_dataloader(self):

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.opt.train.batch_size,
            shuffle=True,
            num_workers=n_cpu
        )
        # END- LOADING TEST DATA********************************

        print("RIKI: TRAIN DATA HAS", len(dataloader))
        return dataloader

    # THE LR AND THE BICUBIC INTERPOLATED HR IMAGE AS INPUT
    # RETURNS THE GENERATED HR OUTPUT
    def forward(self):
        # GETTING THE CURRENT BLEND VALUE
        curr_prog = self.current_epoch + 1
        if self.current_scale_id != 0:
            lo = 0
            hi = self.opt.train.min_epochs[self.current_scale_id]
            self.blend = min((curr_prog - lo) / (hi - lo), 1)
            assert self.blend >= 0

        else:
            self.blend = 1

        # print("SEND LOCATION...", self.input.get_device(), self.label.get_device(), self.interpolated.get_device())
        self.calc_output()
        return self.output

    def calc_output(self):
        self.output = self.net_G(self.input, self.label, upscale_factor=self.model_scale,
                                 blend=self.blend) + self.interpolated

    def set_input(self, lr, hr, bic, label):
        # print("CURRENT DEVICE...",self.device)
        self.input.resize_(lr.size()).copy_(lr)
        self.true_hr.resize_(hr.size()).copy_(hr).to(self.device)
        self.interpolated.resize_(bic.size()).copy_(bic).to(self.device)
        self.label.resize_(label.size()).copy_(label).to(self.device)

        self.input = self.input.to(self.device)
        self.true_hr = self.true_hr.to(self.device)
        self.interpolated = self.interpolated.to(self.device)
        self.label = self.label.to(self.device)
        # self.model_scale = scale

    # GENERATOR LOSS
    def compute_loss(self, xtra_loss):
        self.loss_G = 0
        self.l1_loss = self.l1_criterion(self.output, self.true_hr)
        #self.l1_loss = torch.mean(torch.log(torch.cosh(self.output-self.true_hr)))
        divider = 40*(self.current_scale_id+1)
        deno = divider*divider
        self.loss_G += (self.l1_loss/deno) + 1e-3 * xtra_loss
        #print(1e-3 * xtra_loss, self.loss_G)

    # AT THE END, ADJUST LEARNING RATE IF NECESSARY...LATER
    def on_epoch_end(self):
        time_taken = time() - self.start_time

        logging.info('********************RIKI: EPOCH %d TIME TAKEN: %f NOW: %f' % (self.current_epoch, time_taken, time()))
        curr_prog = self.current_epoch + 1

        self.progress = curr_prog

        self.train_losses = 0.0
        self.train_cnt = 0.0
        self.train_l1 = 0.0

        self.psnrs = 0.0
        self.losses = 0.0
        self.losses_l1 = 0.0
        self.psnr_count = 0.0

        self.start_time = time()

    def on_validation_epoch_end(self) -> None:
        if self.psnr_count > 0:
            avg_psnr = self.psnrs / self.psnr_count
            avg_losses = self.losses / self.psnr_count
            avg_l1 = self.losses_l1/self.psnr_count
        else:
            avg_psnr = 0
            avg_losses = 0
            avg_l1 = 0

        logging.info('RIKI: EPOCH %d PSNRS: %f VALIDATION LOSSES: %s %s' % (self.current_epoch, avg_psnr, str(float(avg_losses)), str(float(avg_l1))))
        '''self.psnrs = 0.0
        self.losses = 0.0
        self.losses_l1 = 0.0
        self.psnr_count = 0.0'''

    def on_epoch_start(self):
        self.start_time = time()


    def training_step(self, batch, batch_nb, optimizer_idx):
        epoch = self.current_epoch

        criterion_GAN = torch.nn.MSELoss()

        # d_decay = self.opt.train.D_lr_decay
        imgs = batch

        # logging.info("RIKI: FILENAME IS "+str(imgs['fname']))

        # print("IMAGE LR SHAPE",imgs['lr'].size())
        self.set_input(imgs['lr'], imgs['hr'], imgs['bicubic'], imgs['tip'])
        bic_image = imgs['bicubic']
        act_img = imgs['hr']
        current_scale = int(imgs['scale'][0].item())

        '''if current_scale != self.model_scale:
            print("DATA/MODEL SCALE:", current_scale,self.model_scale)'''

        current_blend = self.blend

        # GENERATOR MAKING HR IMAGE
        gen_hr = self.forward()

        # DISCRIMINATOR'S VERDICT OF FAKE IMAGE
        discrim_verdict = self.discriminator(gen_hr - bic_image, imgs['tip'], current_scale, current_blend)
        # TWEAKING ********************************************************************
        chance = random.randint(0, 100)

        switcheroo = False
        '''and int(imgs['scale'][0].item()) > 2'''
        if chance < 5:
            switcheroo = True
            # print("SWITCHING ON...")

        if switcheroo:
            fake = torch.ones(discrim_verdict.size(), dtype=torch.float32).to(self.device)
            valid = torch.zeros_like(fake, dtype=torch.float32).to(self.device)
        else:
            valid = torch.ones(discrim_verdict.size(), dtype=torch.float32).to(self.device)
            fake = torch.zeros_like(valid, dtype=torch.float32).to(self.device)

        rand = random.uniform(0, 0.1)
        valid = valid - rand
        fake = fake + rand

        if optimizer_idx == 0:
            # print("OPTIMIZING GEN...")
            # TWEAKING ********************************************************************

            # Adversarial loss
            # GENERATOR NEEDS TO FORCE DISCRIMINATOR TO DEEM FAKE IMAGES AS TRUE
            # print("SEND LOCATION2...", discrim_verdict.get_device(), valid.get_device())
            loss_GAN = criterion_GAN(discrim_verdict, valid)

            # SCALE SWITCHING HAPPENS HERE IF #EPOCHS IS REACHED
            # INCREMENT PROGRESS ONLY ONCE PER EPOCH
            self.compute_loss(loss_GAN)

            self.train_losses += self.loss_G
            self.train_l1 += self.l1_loss
            self.train_cnt += 1
            #print(self.train_l1)
            # WHETHER TO SWITCH SCALES NOW
            # self.ret = self.increment_training_progress()
            if batch_nb % 100 == 0:

                if self.train_cnt > 0:
                    avg_losses = self.train_losses / self.train_cnt
                    avg_l1 = self.train_l1 / self.train_cnt
                else:
                    avg_losses = 0
                    avg_l1 = 0

                self.log('train_loss', self.loss_G, on_step=True, on_epoch=True, logger=True)
                logging.info('RIKI:>>>>>> EPOCH %d TRAINING LOSSES: %f %f' % (self.current_epoch, avg_losses, avg_l1))
                '''self.train_losses = 0.0
                self.train_cnt = 0.0
                self.train_l1 = 0.0'''


            if batch_nb % 100 == 0:
                bic = self.denormalize(self.interpolated[0])
                hr = self.denormalize(self.true_hr[0])
                ghr = self.denormalize(self.output[0])
                op = ImageManipulator.combine_img_list((bic.unsqueeze(0), hr.unsqueeze(0), ghr.unsqueeze(0)), 3)
                fname = self.base_dir + self.opt.xtra.save_path + file_separator + str(self.model_scale) + "_" + str(self.current_epoch)+"_"+str(batch_nb) + ".jpg"
                #print('RIKI: SAVE TO ' + fname)
                save_image(op, fname, normalize=False)

            return {'loss': self.loss_G}

        # **************BEGIN: OPTIMIZING DISCRIMINATOR
        if optimizer_idx == 1:

            # HANDLING SCALE CHANGE FOR DISCRIMINATOR
            # print("OPTIMIZING DIS...")
            gen_hr = self.forward()
            real_eval = self.discriminator(act_img - bic_image, imgs['tip'], current_scale, current_blend)
            fake_eval = self.discriminator(gen_hr.detach() - bic_image, imgs['tip'], current_scale, current_blend)
            # print("SIZES ",real_eval.size(),fake_eval.size())
            # Loss of real and fake images
            # print("SIZESSSSSS1", real_eval.size(), valid.size())
            loss_real = criterion_GAN(real_eval, valid)
            # print(loss_real)
            # loss_fake = criterion_GAN(discriminator(gen_hr.detach().cpu()), fake)
            # print("SIZESSSSSS2", fake_eval.size(), fake.size())
            loss_fake = criterion_GAN(fake_eval, fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            # **************END: OPTIMIZING DISCRIMINATOR

            # dict: eg. 'l1_x2'
            errors = self.get_current_errors()
            for key, item in errors.items():
                # print("ERROR", self.errors_accum[key], type(self.errors_accum[key]), type(item))
                self.errors_accum[key].append(item)

            return {'loss': loss_D}

    def actual_increment(self):
        print('RIKI: trainer TIME TO INCREASE TRAINING SCALE')
        self.current_scale_id += 1

        # RESET LEARNING RATES
        self.lr = self.opt.train.lr
        self.d_learn = self.opt.train.lr * 0.1
        #self.reset_learning_rate()

        self.net_G.current_scale_idx = self.current_scale_id
        self.model_scale *= 2

        if self.current_scale_id == len(self.opt.data.scale):
            print("TRAINING OVER...TIME TO GO....BYE")
            exit(1)

        # SO THAT THE DATA LOADER LOADS THE CORRECT IMAGES
        self.train_dataset.set_scales(self.current_scale_id)
        self.val_dataset.set_scales(self.current_scale_id)
        self.testing_dataset.set_scales(self.current_scale_id)
        # REPEAT self.opt.data.scale[i] FOR self.current_scale_idx + 1 TIMES

        # print("CHANGING....",self.current_scale_id, " ", self.opt.data.scale)
        training_scales = [
            self.opt.data.scale[i]
            for i in range(self.current_scale_id + 1)
        ]

        print('RIKI: UPDATED TRAINING SCALES: {}'.format(str(training_scales)))

    def get_current_errors(self):
        d = OrderedDict()
        if hasattr(self, 'l1_loss'):
            d['l1_x%d' % self.model_scale] = self.l1_loss.item()

        return d

    def set_learning_rate(self, lr, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def reset_learning_rate(self):
        self.set_learning_rate(self.lr, self.optimizer_G)

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = self.d_learn

    def update_learning_rate(self):
        """update learning rate with exponential decay"""
        lr = self.lr * self.opt.train.lr_decay
        if lr < self.opt.train.smallest_lr:
            return
        self.set_learning_rate(lr, self.optimizer_G)
        print('update learning rate: %f -> %f' % (self.lr, lr))
        self.lr = lr

        if self.d_learn > self.opt.train.D_smallest_lr:
            self.d_learn = self.d_learn * self.d_decay
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = self.d_learn

    # SAVING TO MEMORY
    def save(self, epoch, lr, scale, make_save=False):
        #(opt_g, opt_d) = self.optimizers()
        to_save = {
            'network': self.save_network(self.net_G, 'G', str(epoch), str(scale), make_save),
            #'optim': self.save_optimizer(self.optimizer_G, 'G', epoch, lr, str(scale), make_save),
        }

        print("RIKI: SAVED LATEST MODEL %d %d %d...DISK: %s" % (epoch, lr, scale, str(make_save)))
        return to_save

    def save_network(self, network, network_label, epoch_label, scale, make_save=False):
        network = network.module if isinstance(
            network, torch.nn.DataParallel) else network
        # save_filename = '%s_net_%s_x%s.pth' % (epoch_label, network_label, scale)
        save_filename = 'net_%s_x%s.pth' % (network_label, scale)
        save_path = os.path.join(self.save_dir, save_filename)
        to_save = {
            'state_dict': network.state_dict(),
            'path': save_path
        }

        if make_save:
            torch.save(to_save, save_path)

        return to_save

    def save_optimizer(self, optimizer, network_label, epoch, lr, scale, make_save=False):
        save_filename = 'optim_%s_x%s.pth' % (network_label, scale)
        save_path = os.path.join(self.save_dir, save_filename)

        to_save = {
            'state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'lr': lr,
            'path': save_path
        }

        if make_save:
            torch.save(to_save, save_path)

        return to_save

    def load(self, resume_from):
        print("LOADING: ", resume_from[0])
        self.load_network(self.net_G, resume_from[0])
        # self.load_optimizer(self.optimizer_G, resume_from[1])

    def load_network(self, network, saved_path):
        network = network.module if isinstance(
            network, torch.nn.DataParallel) else network
        loaded_state = torch.load(saved_path)['state_dict']
        loaded_param_names = set(loaded_state.keys())

        # allow loaded states to contain keys that don't exist in current model
        # by trimming these keys;
        own_state = network.state_dict()
        extra = loaded_param_names - set(own_state.keys())
        if len(extra) > 0:
            print('Dropping ' + str(extra) + ' from loaded states')
        for k in extra:
            del loaded_state[k]

        try:
            network.load_state_dict(loaded_state)
        except KeyError as e:
            print(e)
        print('RIKI: loaded network state from ' + saved_path)

    def load_optimizer(self, optimizer, saved_path):

        data = torch.load(saved_path)
        loaded_state = data['state_dict']
        optimizer.load_state_dict(loaded_state)

        # Load more params
        self.start_epoch = data['epoch']
        self.lr = data['lr']

        print('RIKI: loaded optimizer state from ' + saved_path)

    # USING G1, GENERATE THE HR' TO BE USED IN G2
    def evaluate_and_generate_for_g2_new(self):
        self.calc_output()
        # CALCULATING L1 LOSS
        self.compute_loss(0)

        my_loss = self.loss_G.item()

        return self.output, my_loss

    # MODEL EVALUATION
    def evaluate(self):

        self.calc_output()
        #self.compute_loss(0.0)

        im1 = self.tensor2im(self.true_hr)
        im2 = self.tensor2im(self.output)

        im3 = self.tensor2im(self.interpolated)

        psnrval, ssim = eval_psnr_and_ssim_old(im1, im2, self.model_scale)
        psnrval_b, ssim_b = eval_psnr_and_ssim_old(im1, im3, self.model_scale)

        return (psnrval, ssim,psnrval_b, ssim_b)

    def evaluate_time(self):
        self.calc_output()
        curr_time = time()
        return curr_time

    def set_eval(self):
        self.net_G.eval()
        self.isTrain = False

    def set_train(self):
        self.net_G.train()
        self.net_G.requires_grad_(True)
        self.discriminator.requires_grad_(True)
        self.isTrain = True

    def reset_eval_result(self):
        for k in self.eval_dict:
            self.eval_dict[k].clear()

    # AVERAGING CURRENT PSNRs FROM eval_dict
    def get_current_eval_result(self):
        eval_result = OrderedDict()
        for k, vs in self.eval_dict.items():
            eval_result[k] = 0
            if vs:
                for v in vs:
                    eval_result[k] += v
                eval_result[k] /= len(vs)
        return eval_result

    def get_current_eval_result_pyr(self):

        return self.eval_dict.copy()

    def update_best_eval_result(self, epoch, current_eval_result=None):
        if current_eval_result is None:
            eval_result = self.get_current_eval_result()
        else:
            eval_result = current_eval_result
        is_best_sofar = any(
            [np.round(eval_result[k], 2) > np.round(v, 2) for k, v in self.best_eval.items()])
        # print("RIKI: trainer IS BEST SO FAR: "+str(is_best_sofar))
        if is_best_sofar:
            self.best_epoch = epoch
            self.best_eval = {
                k: max(self.best_eval[k], eval_result[k])
                for k in self.best_eval
            }

