import os
import os.path as osp
import sys
import yaml
from utils.dotdict import DotDict
from model.trainer_one_shot import HierarchicalSRTrainer
from pytorch_lightning import Trainer
import socket
import logging
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import shutil
import pickle

hostname=str(socket.gethostname())
base_path="/s/"+hostname+"/a/nobackup/galileo/sapmitra/SRImages"
# HOW OFTEN IN AN EPOCH TO VALIDATE
validation_freq  = 0.25

configFile="/s/chopin/b/grad/sapmitra/Glance/config/prosrgan.yaml" #Config File
ig_file="/s/chopin/b/grad/sapmitra/Glance/config/ignorables.txt"
n_cpu=8 #number of cpu threads to use during batch generation
file_separator = "/"

def clear_old(params):
    old_log = base_path + '/glance_' + hostname + '.log'
    if os.path.exists(old_log):
        print("REMOVED OLD LOG")
        os.remove(old_log)
    clear_or_create(base_path + params.xtra.save_path)
    #clear_or_create(base_path + params.xtra.saved_images)


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
    clear_old(args)
    clear_or_create(base_path+args.xtra.chkpt_path)
    clear_or_create(base_path + '/glance_lightning_' + hostname)
    flush_steps = 20

    ignorables = []
    with open(ig_file, 'rb') as f:
        my_list = pickle.load(f)
        print(len(my_list))

        if len(my_list) > 0:
            ignorables.extend(my_list)

    args.ignorables = ignorables

    tb_logger = pl_loggers.TensorBoardLogger(base_path + '/glance_lightning_' + hostname)
    logging.basicConfig(filename=base_path + '/glance_' + hostname + '.log', level=logging.INFO)
    print("BASE PATH: " + base_path)
    print("LOG PATH:", (base_path + '/glance_' + hostname + '.log'))

    # checkpoint is the directory where checkpoints are read from and stored
    trainer_model = HierarchicalSRTrainer(opt=args, base_dir=base_path, save_dir=base_path+args.xtra.out_path)

    #********************** TRAINING x2 *********************************
    current_scale_id = 0
    early_stop = EarlyStopping(monitor='val_loss', patience = args.train.training_shutdown_patience, strict=False, verbose=True, mode='min')
    #chk_name = 'glance-{epoch:02d}-{val_loss:.2f}'
    chk_name = 'glance-'+str(current_scale_id)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath=base_path+args.xtra.chkpt_path, filename=chk_name, mode='min')
    #trainer = Trainer(gpus=1,num_nodes=2, max_epochs=args.train.epochs, distributed_backend='ddp')
    trainer = Trainer(gpus=1, val_check_interval=validation_freq, logger=tb_logger, flush_logs_every_n_steps=flush_steps, callbacks=[early_stop, checkpoint_callback],
                      max_epochs=args.train.max_epochs[current_scale_id],
                      min_epochs=args.train.min_epochs[current_scale_id])
    trainer.fit(trainer_model)
    print("*******************************RIKI: FINISHED x2 TRAINING...")
    # IF SAVING IS NECESSARY:
    # trainer_model.save(0, 0, 0, True)

    # ********************** TRAINING x4 *********************************
    current_scale_id = 0
    filename = base_path+args.xtra.chkpt_path+"/"+chk_name + ".ckpt"
    trainer_model = HierarchicalSRTrainer(args, base_dir=base_path, save_dir=base_path + args.xtra.out_path)

    kwargs = {"opt": args, "base_dir":base_path, "save_dir":base_path + args.xtra.out_path, "current_scale": current_scale_id, "isret": True}
    trainer_model = trainer_model.load_from_checkpoint(filename, **kwargs)
    print("RIKI: SUCCESSFULLY LOADED FROM ", filename)
    trainer_model.current_scale_id = current_scale_id
    trainer_model.ret = True
    early_stop = EarlyStopping(monitor='val_loss', patience = args.train.training_shutdown_patience, strict=False, verbose=True, mode='min')
    chk_name = 'glance-' + str(current_scale_id+1)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=base_path + args.xtra.chkpt_path, filename=chk_name, mode='min')
    trainer = Trainer(gpus=1, val_check_interval=validation_freq, logger=tb_logger, flush_logs_every_n_steps=flush_steps, callbacks=[early_stop, checkpoint_callback],
                      max_epochs=args.train.max_epochs[current_scale_id+1], min_epochs=args.train.min_epochs[current_scale_id+1])
    trainer.fit(trainer_model)
    print("*******************************RIKI: FINISHED x4 TRAINING...")
    # IF SAVING IS NECESSARY:
    # trainer_model.save(0, 0, 0, True)

    # ********************** TRAINING x8 *********************************
    current_scale_id = 1
    filename = base_path + args.xtra.chkpt_path + "/" + chk_name + ".ckpt"
    kwargs = {"opt": args, "base_dir": base_path, "save_dir": base_path + args.xtra.out_path, "current_scale": current_scale_id, "isret": True}
    trainer_model = HierarchicalSRTrainer.load_from_checkpoint(filename, **kwargs)
    print("RIKI: SUCCESSFULLY LOADED FROM ", filename)
    trainer_model.current_scale_id = current_scale_id
    trainer_model.ret = True
    early_stop = EarlyStopping(monitor='val_loss', patience = args.train.training_shutdown_patience, strict=False, verbose=True, mode='min')
    chk_name = 'glance-' + str(current_scale_id+1)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=base_path + args.xtra.chkpt_path,
                                          filename=chk_name, mode='min')
    trainer = Trainer(gpus=1, val_check_interval=validation_freq, logger=tb_logger, flush_logs_every_n_steps=flush_steps, callbacks=[early_stop, checkpoint_callback],
                      max_epochs=args.train.max_epochs[current_scale_id+1], min_epochs=args.train.min_epochs[current_scale_id+1])
    trainer.fit(trainer_model)
    print("*******************************RIKI: FINISHED x8 TRAINING...")
    # IF SAVING IS NECESSARY:
    # trainer_model.save(0, 0, 0, True)


if __name__ == '__main__':

    with open(configFile) as file:
        try:
            params = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(0)

    params = DotDict(params)

    if not osp.isdir(base_path+params.xtra.out_path):
        os.makedirs(base_path+params.xtra.out_path)

    #SAVING PARAMETERS FOR RESTART....NEEDS WORK
    #np.save(osp.join(params.ip.out_path, 'params'), params)

    experiment_id = osp.basename(params.xtra.out_path)

    print('experiment ID: {}'.format(experiment_id))

    #pprint(params)
    modelling(params)
