import math
import os
import os.path as osp
import torch
import numpy as np
from utils.im_manipulation import tensor2im,eval_psnr_and_ssim
from model.gen_dis_inpaint import HierarchicalGenerator,HierarchicalDiscriminator
from data.ImageDataset_G2_inpaint import ImageDataset_G2_i
import utils.im_manipulation as ImageManipulator
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl
from collections import OrderedDict,defaultdict
import random
from torchvision import transforms
from utils.Phase import Phase
from torch.utils.data import DataLoader
import socket
import logging
from time import time,sleep

n_cpu = 8
file_separator = "/"
num_nodes=15
file_base='/s/chopin/b/grad/sapmitra/status/'
# MODULE THAT HOUSES THE GENERATOR AND DISCRIMINATOR
class HierarchicalSRTrainer(pl.LightningModule):

    # INIT OF GENERATOR AND DISCRIMINATOR
    def __init__(self,opt, base_dir = "/nopath",save_dir="data/checkpoints"):
        super(HierarchicalSRTrainer, self).__init__()

        self.g1_input = torch.zeros(opt.train.batch_size, 3, 48, 48, dtype=torch.float32)
        self.true_hr = torch.zeros_like(self.g1_input, dtype=torch.float32)
        self.composite = torch.zeros_like(self.true_hr, dtype=torch.float32)
        self.mask = torch.zeros_like(self.true_hr, dtype=torch.long)

        self.ret = False
        self.base_dir = base_dir

        tile_dir = opt.xtra.img_tile_path

        hostname = str(socket.gethostname())
        self.hostname = hostname
        self.ip_dir = tile_dir.replace("HOSTNAME", hostname, 1)
        self.albums = opt.xtra.albums

        self.host_id = int(hostname.split(".")[0].split("-")[1]) - 177
        self.opt = opt
        self.save_dir = save_dir
        std = np.asarray(self.opt.train.dataset.stddev)
        mean = np.asarray(self.opt.train.dataset.mean)
        self.denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

        # NUMBER OF BATCHES POSSIBLE IN 1 RUN OF DATASET
        self.start_epoch = 0
        self.progress = self.start_epoch / opt.train.epochs
        self.blend = 1
        # INDEX OF THE CURRENT MODEL SCALE
        self.current_scale_id=0
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

        self.lr = self.opt.train.lr

        self.d_learn = self.opt.train.lr * 0.1
        self.d_decay = self.opt.train.D_lr_decay
        self.discriminator = HierarchicalDiscriminator(self.opt.Discriminator.num_op_v,
                                                  self.opt.Discriminator.level_config,
                                                  self.opt.data.scale[-1],
                                                  self.opt.G.num_ip_channels, self.opt.G.lr_res,
                                                  self.opt.G.num_classes)

        self.l1_criterion = torch.nn.L1Loss()
        self.errors_accum = defaultdict(list)

        self.start_time = time()

    def configure_optimizers(self):
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

        return [self.optimizer_G,self.optimizer_D],[]


    def train_dataloader(self) :
        training_dataset = ImageDataset_G2_i(phase=Phase.TRAIN,
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
                                             mask_grid=self.opt.edge_param.mask_grid_size)

        dataloader = DataLoader(
            training_dataset,
            batch_size=self.opt.train.batch_size,
            shuffle=True,
            num_workers=n_cpu,
        )

        self.testing_dataset = ImageDataset_G2_i(phase=Phase.TEST,
                                                 albums=self.albums,
                                                 img_dir=self.ip_dir,
                                                 img_type=self.opt.xtra.img_type,
                                                 mean=self.opt.test.train_dataset.mean,
                                                 stddev=self.opt.train.dataset.stddev,
                                                 scales=self.opt.data.scale,
                                                 high_res=self.opt.xtra.high_res,
                                                 num_inputs=self.opt.xtra.num_inputs,
                                                 num_tests= self.opt.xtra.num_tests,
                                                 pyramid_levels=self.opt.xtra.pyramid_levels,
                                                 mask_grid=self.opt.edge_param.mask_grid_size,
                                                 ignore_files=training_dataset.image_fnames)

        self.testing_data_loader = DataLoader(
            self.testing_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=n_cpu,
        )
        # END- LOADING TEST DATA********************************
        # NUMBER OF BATCHES POSSIBLE IN 1 RUN OF DATASET
        self.dataLen = len(dataloader)
        self.dataset = training_dataset
        print("RIKI: TRAIN DATA HAS", len(dataloader))
        print("RIKI: TEST DATA HAS", len(self.testing_data_loader))
        return dataloader


    # THE LR AND THE BICUBIC INTERPOLATED HR IMAGE AS INPUT
    # RETURNS THE GENERATED HR OUTPUT
    def forward(self):
        # GETTING THE CURRENT BLEND VALUE
        curr_prog = (self.current_epoch + 1) / self.opt.train.epochs

        if self.current_scale_id != 0:
            lo, hi = self.opt.train.growing_steps[self.current_scale_id * 2 - 2 : self.current_scale_id * 2]
            #self.blend = min((self.progress - lo) / (hi - lo), 1)
            self.blend = min((curr_prog - lo) / (hi - lo), 1)
            assert self.blend >= 0
            if self.blend > 1:
                logging.warning("BLEND HAS GONE TEMPORARILY ABOVE 1")
                self.blend = 1
        else:
            self.blend = 1

        #print("SEND LOCATION...", self.input.get_device(), self.label.get_device(), self.interpolated.get_device())
        self.calc_output()
        return self.output

    def calc_output(self):
        op = self.net_G(self.g1_input, upscale_factor=self.model_scale, blend=self.blend)
        self.output = op*(1 - self.mask) + self.composite

    def set_input(self, g1_input, hr, composite, mask):
        #print("CURRENT DEVICE...",self.device)
        self.g1_input.resize_(g1_input.size()).copy_(g1_input)
        self.true_hr.resize_(hr.size()).copy_(hr).to(self.device)
        self.composite.resize_(composite.size()).copy_(composite).to(self.device)
        self.mask.resize_(mask.size()).copy_(mask).to(self.device)

        self.g1_input = self.g1_input.to(self.device)
        self.true_hr = self.true_hr.to(self.device)
        self.composite = self.composite.to(self.device)
        self.mask = self.mask.to(self.device)
        #self.model_scale = scale

    # GENERATOR LOSS
    def compute_loss(self, xtra_loss):
        self.loss_G = 0
        inv_mask = 1 - self.mask
        self.l1_loss = self.l1_criterion(self.output*inv_mask, self.true_hr*inv_mask) + 1e-3 * xtra_loss
        self.loss_G += self.l1_loss

    # AT THE END, ADJUST LEARNING RATE IF NECESSARY...LATER
    def on_epoch_end(self):
        time_taken = time() - self.start_time

        #SET FILENAMES AND ITERATE
        if len(self.testing_data_loader)>0:
            self.testing_dataset.copy_scales(self.dataset.current_scale)
            #self.testing_dataset.set_scales(self.dataset.current_scale)
            with torch.no_grad():

                # use validation set
                self.set_eval()

                self.reset_eval_result()

                total_loss = 0.0
                total_cnt = 0
                #print("RIKI: CHECK "+str(test_dataset.current_scale))
                for i, imgs in enumerate(self.testing_data_loader):
                    self.set_input(imgs['g1_input'], imgs['hr'], imgs['composite'], imgs['mask'])
                    #PSNR COMPUTATIONS HERE
                    l1 = self.evaluate()
                    if l1 >= 0:
                        total_loss += l1
                        total_cnt += 1

                test_result = self.get_current_eval_result()
                l1_result = 99.0
                if total_cnt>0:
                    l1_result = total_loss / total_cnt

                epoch = self.current_epoch
                self.update_best_eval_result(epoch, test_result)

                logging.info(
                    'RIKI: EVAL AT EPOCH %d TIME TAKEN %f L1 LOSS %f: ' % (epoch, time_taken, l1_result) + ' | '.join([
                        '{}: {:.07f}'.format(k, v)
                        for k, v in test_result.items()
                    ]))

                '''print(
                    'RIKI: best so far %d : ' % self.best_epoch + ' | '.join([
                        '{}: {:.07f}'.format(k, v)
                        for k, v in self.best_eval.items()
                    ]))'''

                ################# update learning rate  #################
                # IF THE BEST EPOCH WAS SEEN A WHILE BACK...IT'S TIME TO UPDATE THE LEARNING RATE BY 1/2
                if (epoch - self.best_epoch) > self.opt.train.lr_schedule_patience:
                    # trainer.save('last_lr_%g' % trainer.lr, epoch, trainer.lr)
                    logging.info("RIKI: UPDATING LR..."+str(self.d_learn)+" "+str(self.opt.train.D_smallest_lr))
                    self.update_learning_rate()

                    if self.d_learn > self.opt.train.D_smallest_lr:
                        self.d_learn = self.d_learn * self.d_decay
                        for param_group in self.optimizer_D.param_groups:
                            param_group['lr'] = self.d_learn
                        #print('update discriminator learning rate: %f' % (d_learn))
                self.set_train()

            curr_prog = (self.current_epoch + 1) / self.opt.train.epochs
            """increment self.progress and D, G scale_idx"""
            # 1/2/3 = 1/(2*3) = .166
            # PITSTOPS AT 12%, 25%, 45%, 60%, 100%
            self.progress = curr_prog

            # print("CURRENT PROGRESS ",self.progress," ",self.opt.train.growing_steps[self.current_scale_id * 2]," ",self.opt.data.scale )
            # IF THE #EPOCHS HAS HIT A BREAK-POINT - THIS MEANS A SCALE CHANGE IS TO BE INITIATED
            if self.progress > self.opt.train.growing_steps[self.current_scale_id * 2]:
                # print("HERE", self.current_scale_id, len(self.opt.data.scale))
                if self.current_scale_id < len(self.opt.data.scale):
                    self.ret = True

            if self.ret:
                self.ret = False
                logging.info("SWITCHING GEARS...RESTARTING EPOCH " + str(self.current_epoch))
                self.actual_increment()
                self.set_train()

            if self.current_epoch == self.opt.train.epochs - 1:
                print("RIKI: SAVING FINAL MODEL")
                self.save(self.current_epoch, self.lr, self.current_scale_id, True)


    def on_epoch_start(self):
        self.start_time = time()

    '''def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:

        print("FINISHED......")'''

    '''def on_batch_start(self):
        print("BATCH STARTING")'''

    def training_step(self, batch, batch_nb, optimizer_idx):
        epoch = self.current_epoch
        criterion_GAN = torch.nn.MSELoss()

        imgs = batch

        #print("IMAGE LR SHAPE",imgs['lr'].size())
        #self.set_input(imgs['lr'], imgs['hr'], imgs['bicubic'], imgs['tip'])
        self.set_input(imgs['g1_input'], imgs['hr'], imgs['composite'], imgs['mask'])
        bic_image = imgs['bicubic']
        act_img = imgs['hr']
        current_scale = int(imgs['scale'][0].item())

        '''if current_scale != self.model_scale:
            print("DATA/MODEL SCALE:", current_scale,self.model_scale)'''

        current_blend = self.blend

        # GENERATOR MAKING HR IMAGE
        gen_hr = self.forward()

        # DISCRIMINATOR'S VERDICT OF FAKE IMAGE
        #print("SEND LOCATION2...", gen_hr.get_device(), bic_image.get_device(), imgs['tip'].get_device())
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
            #print("OPTIMIZING GEN...")
            # TWEAKING ********************************************************************

            # Adversarial loss
            # GENERATOR NEEDS TO FORCE DISCRIMINATOR TO DEEM FAKE IMAGES AS TRUE
            #print("SEND LOCATION2...", discrim_verdict.get_device(), valid.get_device())
            loss_GAN = criterion_GAN(discrim_verdict, valid)

            # SCALE SWITCHING HAPPENS HERE IF #EPOCHS IS REACHED
            # INCREMENT PROGRESS ONLY ONCE PER EPOCH
            self.compute_loss(loss_GAN)

            # WHETHER TO SWITCH SCALES NOW
            #self.ret = self.increment_training_progress()

            #print("OPTIMIZING GEN OVER...")
            return {'loss': self.loss_G}

        # **************BEGIN: OPTIMIZING DISCRIMINATOR
        if optimizer_idx == 1:

            # HANDLING SCALE CHANGE FOR DISCRIMINATOR
            #print("OPTIMIZING DIS...")
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
                #print("ERROR", self.errors_accum[key], type(self.errors_accum[key]), type(item))
                self.errors_accum[key].append(item)

            tip_val = imgs['tip'][0].item()
            # HOW OFTEN TO PRINT ERROR STATUS TO CONSOLE: print_errors_freq
            '''if batch_nb % self.opt.train.io.print_errors_freq == 0:
                for key, item in errors.items():
                    if len(self.errors_accum[key]):
                        self.errors_accum[key] = np.nanmean(self.errors_accum[key])
                message = 'RIKI: EPOCH:%d BATCH: %d SCALE: %d PYR_HEAD: %s BLEND: %f ERRORS: %s %s %s %s' % (
                epoch, batch_nb, int(imgs['scale'][0].item()), str(tip_val), self.blend, str(self.errors_accum[key]),
                str(loss_D.item()), str(loss_real.item()), str(loss_fake.item()))
                print(message)
                #errors_accum_prev = errors_accum
                # t = time() - iter_start_time
                # iter_start_time = time()
                self.errors_accum = defaultdict(list)

            # SAVING INTERMEDIATE IMAGE RESULTS
            if batch_nb % self.opt.train.io.save_img_freq == 0:

                key = 'l1_x' + str(self.model_scale)

                lr = self.denormalize(imgs['lr'][0])
                hr = self.denormalize(imgs['hr'][0])
                ghr = self.denormalize(gen_hr[0])

                op = ImageManipulator.combine_images(lr.unsqueeze(0), hr.unsqueeze(0), ghr.unsqueeze(0),
                                                     self.model_scale)
                fname = self.base_dir + self.opt.xtra.save_path + file_separator + "opimg_x" + str(
                    self.model_scale) + "_" + str(tip_val) + "_" + str(epoch) + "_" + str(
                    batch_nb) + ".jpg"
                print('RIKI: SAVE TO ' + fname)
                save_image(op, fname, normalize=False)
            '''
            #print("OPTIMIZING DIS OVER...")
            return {'loss': loss_D}

    def reset_learning_rate(self):
        self.set_learning_rate(self.lr, self.optimizer_G)

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = self.d_learn

    def actual_increment(self):
        # SAVING CURRENT BEST
        print('RIKI: STORING CURRENT BEST MODEL....DUE TO SCALE UPDATE ')

        # SAVING TRAINED MODEL TO DISK
        self.save(self.current_epoch, self.lr, self.current_scale_id, True)

        print('RIKI: trainer TIME TO INCREASE TRAINING SCALE')
        self.current_scale_id += 1

        self.net_G.current_scale_idx = self.current_scale_id
        self.model_scale *= 2

        #RESET LEARNING RATES
        self.lr = self.opt.train.lr
        self.d_learn = self.opt.train.lr * 0.1
        self.reset_learning_rate()

        if self.current_scale_id == len(self.opt.data.scale):
            print("TIME TO GO....BYE")
            exit(1)

        # SO THAT THE DATA LOADER LOADS THE CORRECT IMAGES
        self.dataset.set_scales(self.current_scale_id)
        # REPEAT self.opt.data.scale[i] FOR self.current_scale_idx + 1 TIMES

        # print("CHANGING....",self.current_scale_id, " ", self.opt.data.scale)
        training_scales = [
            self.opt.data.scale[i]
            for i in range(self.current_scale_id + 1)
        ]

        print('RIKI: UPDATED TRAINING SCALES: {}'.format(str(training_scales)))

    # THIS HELPS JUMP THE SCALE DURING TRAINING
    '''def increment_training_progress(self):
        ret = False
        """increment self.progress and D, G scale_idx"""
        # 1/2/3 = 1/(2*3) = .166
        # PITSTOPS AT 12%, 25%, 45%, 60%, 100%
        self.progress += 1 / self.dataLen / self.opt.train.epochs

        #print("CURRENT PROGRESS ",self.progress," ",self.opt.train.growing_steps[self.current_scale_id * 2]," ",self.opt.data.scale )
        # IF THE #EPOCHS HAS HIT A BREAK-POINT - THIS MEANS A SCALE CHANGE IS TO BE INITIATED
        if self.progress > self.opt.train.growing_steps[self.current_scale_id * 2]:
            #print("HERE", self.current_scale_id, len(self.opt.data.scale))
            if self.current_scale_id < len(self.opt.data.scale):
                ret = True

        return ret'''


    def get_current_errors(self):
        d = OrderedDict()
        if hasattr(self, 'l1_loss'):
            d['l1_x%d' % self.model_scale] = self.l1_loss.item()

        return d

    def set_learning_rate(self, lr, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_learning_rate(self):
        """update learning rate with exponential decay"""
        lr = self.lr * self.opt.train.lr_decay
        if lr < self.opt.train.smallest_lr:
            return
        self.set_learning_rate(lr, self.optimizer_G)
        print('update learning rate: %f -> %f' % (self.lr, lr))
        self.lr = lr

    # SAVING TO MEMORY
    def save(self, epoch, lr, scale, make_save=False):
        (opt_g, opt_d) = self.optimizers()
        to_save = {
            'network': self.save_network(self.net_G, 'G', str(epoch), str(scale), make_save),
            'optim': self.save_optimizer(opt_g, 'G', epoch, lr, str(scale), make_save),
        }

        print("RIKI: SAVED LATEST MODEL %d %d %d...DISK: %s"%(epoch, lr, scale, str(make_save)))
        return to_save

    def save_network(self, network, network_label, epoch_label, scale, make_save=False):
        network = network.module if isinstance(
            network, torch.nn.DataParallel) else network
        #save_filename = '%s_net_%s_x%s.pth' % (epoch_label, network_label, scale)
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
        self.load_network(self.net_G, resume_from[0])
        #self.load_optimizer(self.optimizer_G, resume_from[1])

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

    # MODEL EVALUATION
    def evaluate(self):

        self.calc_output()
        self.compute_loss(0)

        im1 = self.tensor2im(self.true_hr*(1 - self.mask))
        im2 = self.tensor2im(self.output*(1 - self.mask))

        psnrval = eval_psnr_and_ssim(im1, im2, self.model_scale)[0]
        if not math.isfinite(psnrval):
            logging.info("RIKI: ENCOUNTERED INF...IGNORE IT")
        else:
            eval_res = {
                'psnr_x%d' % (self.model_scale): psnrval
                    #eval_psnr_and_ssim(im1, im2, self.model_scale)[0]
            }
            for k, v in eval_res.items():
                self.eval_dict[k].append(v)
            return self.loss_G

        return -7.0

    def set_eval(self):
        self.net_G.eval()
        self.isTrain = False

    def set_train(self):
        self.net_G.train()
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
            [np.round(eval_result[k],2) > np.round(v,2) for k, v in self.best_eval.items()])
        #print("RIKI: trainer IS BEST SO FAR: "+str(is_best_sofar))
        if is_best_sofar:
            self.best_epoch = epoch
            self.best_eval = {
                k: max(self.best_eval[k], eval_result[k])
                for k in self.best_eval
            }
