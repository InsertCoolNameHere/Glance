import os
import os.path as osp
import sys
import yaml
from utils.dotdict import DotDict
from model.trainer_inpaint_one_shot import HierarchicalSRTrainer
import socket
from pytorch_lightning import Trainer
import logging
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import shutil
import pickle
from time import time

hostname=str(socket.gethostname())
base_path="/s/"+hostname+"/a/nobackup/galileo/sapmitra/SRImages"
configFile="/s/chopin/b/grad/sapmitra/Glance/config/prosrgan_G2.yaml" #Config File
ig_file="/s/chopin/b/grad/sapmitra/Glance/config/ignorables.txt"
n_cpu=1 #number of cpu threads to use during batch generation
file_separator = "/"
validation_freq = 0.25
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["NCCL_SOCKET_IFNAME"]="eno1"


def clear_old(params):
    old_log = base_path + '/glance2_' + hostname + '.log'
    if os.path.exists(old_log):
        print("REMOVED OLD LOG")
        os.remove(old_log)
    clear_or_create(base_path + params.xtra.save_path)

def clear_or_create(p):
    if not osp.isdir(p):
        os.makedirs(p)
    else:
        # REMOVE ENTRIES
        for f in os.listdir(p):
            if not osp.isdir(os.path.join(p, f)):
                os.remove(os.path.join(p, f))
            else:
                shutil.rmtree(os.path.join(p, f))

def modelling(args):
    num_nodes = 24
    clear_old(args)
    clear_or_create(base_path + args.xtra.chkpt_path)
    # REMOVE OLD TENSORBOARD LOG
    clear_or_create(base_path + '/glance2_lightning_' + hostname)
    flush_steps = 100


    ignorables = []
    with open(ig_file, 'rb') as f:
        my_list = pickle.load(f)
        print(len(my_list))

        if len(my_list) > 0:
            ignorables.extend(my_list)

    args.ignorables = ignorables


    tb_logger = pl_loggers.TensorBoardLogger(base_path + '/glance2_lightning_' + hostname)
    logging.basicConfig(filename=base_path + '/glance2_' + hostname + '.log', level=logging.INFO)
    print("BASE PATH: " + base_path)
    print("LOG PATH:", (base_path + '/glance_' + hostname + '.log'))

    # checkpoint is the directory where checkpoints are read from and stored
    trainer_model = HierarchicalSRTrainer(opt = args, base_dir=base_path, save_dir=base_path + args.xtra.chkpt_path)

    # ********************** TRAINING x2 *********************************
    current_scale_id = 0
    early_stop = EarlyStopping(monitor='val_loss', patience=args.train.training_shutdown_patience, strict=False, verbose=True, mode='min')
    # chk_name = 'glance-{epoch:02d}-{val_loss:.2f}'
    chk_name = 'glance-' + str(current_scale_id)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=base_path + args.xtra.chkpt_path,
                                          filename=chk_name, mode='min')
    # trainer = Trainer(gpus=1,num_nodes=2, max_epochs=args.train.epochs, distributed_backend='ddp')
    trainer = Trainer(gpus=1, num_nodes=num_nodes, val_check_interval=validation_freq, logger=tb_logger, flush_logs_every_n_steps=flush_steps, callbacks=[early_stop, checkpoint_callback],
                      max_epochs=args.train.max_epochs[current_scale_id], min_epochs=args.train.min_epochs[current_scale_id],
                      distributed_backend='ddp')
    start_time = time()
    trainer.fit(trainer_model)
    logging.info("**********************************DONE x2**********************" + str(time() - start_time))
    print("*******************************RIKI: FINISHED x2 TRAINING...SLEEPING")
    trainer_model.save(0, 0, 0, True)

    trainer_model.actual_increment()
    trainer_model.set_train()
    early_stop = EarlyStopping(monitor='val_loss', patience=args.train.training_shutdown_patience, strict=False, verbose=True, mode='min')
    chk_name = 'glance-' + str(current_scale_id + 1)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=base_path + args.xtra.chkpt_path, filename=chk_name, mode='min')
    trainer = Trainer(gpus=1, num_nodes=num_nodes, val_check_interval=validation_freq, logger=tb_logger, flush_logs_every_n_steps=flush_steps, callbacks=[early_stop, checkpoint_callback],
                      max_epochs=args.train.max_epochs[current_scale_id + 1], min_epochs=args.train.min_epochs[current_scale_id + 1],
                      distributed_backend='ddp')
    start_time = time()
    trainer.fit(trainer_model)
    logging.info("**********************************DONE x4**********************"+ str(time() - start_time))
    print("*******************************RIKI: FINISHED x4 TRAINING...SLEEPING")
    trainer_model.save(0, 0, 1, True)

    trainer_model.actual_increment()
    trainer_model.set_train()
    early_stop = EarlyStopping(monitor='val_loss', patience=args.train.training_shutdown_patience, strict=False, verbose=True, mode='min')
    chk_name = 'glance-' + str(current_scale_id + 1)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=base_path + args.xtra.chkpt_path, filename=chk_name, mode='min')
    trainer = Trainer(gpus=1, num_nodes=num_nodes, val_check_interval=validation_freq, logger=tb_logger, flush_logs_every_n_steps=flush_steps, callbacks=[early_stop, checkpoint_callback],
                      max_epochs=args.train.max_epochs[current_scale_id + 1], min_epochs=args.train.min_epochs[current_scale_id + 1],
                      distributed_backend='ddp')
    start_time = time()
    trainer.fit(trainer_model)
    logging.info("**********************************DONE x8**********************" + str(time() - start_time))
    print("*******************************RIKI: FINISHED x8 TRAINING...SLEEPING")
    trainer_model.save(0, 0, 2, True)


if __name__ == '__main__':

    with open(configFile) as file:
        try:
            params = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(0)

    params = DotDict(params)

    if not osp.isdir(base_path + params.xtra.out_path):
        os.makedirs(base_path + params.xtra.out_path)

    #SAVING PARAMETERS FOR RESTART....NEEDS WORK
    #np.save(osp.join(params.ip.out_path, 'params'), params)
    if len(sys.argv) > 1:
        num_train = int(sys.argv[1])
        num_val_test = int(sys.argv[2])
        params.xtra.num_inputs = num_train
        params.xtra.num_tests = num_val_test
        params.xtra.num_vals = num_val_test

    experiment_id = osp.basename(params.edge_param.out_path)

    print('experiment ID: {}'.format(experiment_id))

    modelling(params)