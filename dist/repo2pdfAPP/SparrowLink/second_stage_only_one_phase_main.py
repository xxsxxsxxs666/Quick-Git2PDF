from monai.utils import first, set_determinism
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, ITKReader, PersistentDataset
from monai.config import print_config
from monai.metrics.meandice import DiceMetric
import torch
import matplotlib.pyplot as plt
import os
import argparse
import monai
import logging
from utils.Config import Config
import sys
import time
from data.loader import prepare_datalist, prepare_datalist_with_file, \
    prepare_image_list, save_json, write_data_reference, load_json, prepare_main_with_img_datalist_with_file
from transform.utils import get_transform, get_second_stage_only_one_phase
from utils.test import cardio_vessel_segmentation_test, cardio_vessel_segmentation_multi_phase_with_image_test
from utils.trainer import cardio_vessel_segmentation_multi_phase_with_image_train
from utils.inferer import cardio_vessel_segmentation_infer, cardio_vessel_segmentation_multi_phase_with_image_infer
import pathlib
import torch.nn as nn
from torch.distributions.normal import Normal
# print_config()


torch.multiprocessing.set_sharing_strategy('file_system')


def load_weight_from_coarse_segmentation(in_channels, model, weight_path, net_architecture):
    """
    it will change
    """
    in_channels = model.in_channels if hasattr(model, "in_channels") else in_channels
    if net_architecture == 'UNet':
        # d = torch.load(
        #     './experiments/Graduate_project/multi_phase/pretrain/ResUnet/checkpoint/best_metric_model.pth')
        d = torch.load(weight_path)
        shape = list(d['model.0.conv.unit0.conv.weight'].shape)
        shape[1] = in_channels - 1
        if d.get('model.0.residual.weight'):
            noise = nn.Parameter(Normal(0, 1e-5).sample(shape)).to(d['model.0.residual.weight'].device)
            d['model.0.residual.weight'] = torch.cat((d['model.0.residual.weight'], noise), dim=1)
        noise = nn.Parameter(Normal(0, 1e-5).sample(shape)).to(d['model.0.conv.unit0.conv.weight'].device)
        d['model.0.conv.unit0.conv.weight'] = torch.cat((d['model.0.conv.unit0.conv.weight'], noise), dim=1)
        model.load_state_dict(d)

    elif net_architecture == "CS2net":
        d = torch.load(weight_path)
        shape = list(d['encoder1.conv1.weight'].shape)
        shape[1] = in_channels - 1
        noise = nn.Parameter(Normal(0, 1e-5).sample(shape)).to(d['enc_input.conv1.weight'].device)
        d['enc_input.conv1.weight'] = torch.cat((d['enc_input.conv1.weight'], noise), dim=1)
        model.load_state_dict(d)

    elif net_architecture == "SkipDenseUnet":
        d = torch.load(weight_path)
        shape = list(d['features.conv0.weight'].shape)
        shape[1] = in_channels - 1
        noise = nn.Parameter(Normal(0, 1e-5).sample(shape)).to(d['features.conv0.weight'].device)
        d['features.conv0.weight'] = torch.cat((d['features.conv0.weight'], noise), dim=1)
        model.load_state_dict(d)

    elif net_architecture == "SwinTransformerSys3D":
        d = torch.load(weight_path)
        shape = list(d['patch_embed.proj.weight'].shape)
        shape[1] = in_channels - 1
        noise = nn.Parameter(Normal(0, 1e-5).sample(shape)).to(d['patch_embed.proj.weight'].device)
        d['patch_embed.proj.weight'] = torch.cat((d['patch_embed.proj.weight'], noise), dim=1)
        model.load_state_dict(d)

    elif net_architecture == "nnunetv2":
        all_information = torch.load(weight_path)
        d = all_information['network_weights']
        shape = list(d['encoder.stages.0.0.convs.0.conv.weight'].shape)
        shape[1] = in_channels - 1
        noise = nn.Parameter(Normal(0, 1e-5).sample(shape)).to(d['encoder.stages.0.0.convs.0.conv.weight'].device)
        d['encoder.stages.0.0.convs.0.conv.weight'] = torch.cat((d['encoder.stages.0.0.convs.0.conv.weight'], noise), dim=1)
        noise = nn.Parameter(Normal(0, 1e-5).sample(shape)).to(d['encoder.stages.0.0.convs.0.all_modules.0.weight'].device)
        d['encoder.stages.0.0.convs.0.all_modules.0.weight'] = torch.cat((d['encoder.stages.0.0.convs.0.all_modules.0.weight'], noise), dim=1)
        noise = nn.Parameter(Normal(0, 1e-5).sample(shape)).to(d['decoder.encoder.stages.0.0.convs.0.conv.weight'].device)
        d['decoder.encoder.stages.0.0.convs.0.conv.weight'] = torch.cat((d['decoder.encoder.stages.0.0.convs.0.conv.weight'], noise), dim=1)
        noise = nn.Parameter(Normal(0, 1e-5).sample(shape)).to(d['decoder.encoder.stages.0.0.convs.0.all_modules.0.weight'].device)
        d['decoder.encoder.stages.0.0.convs.0.all_modules.0.weight'] = torch.cat((d['decoder.encoder.stages.0.0.convs.0.all_modules.0.weight'], noise), dim=1)
        model.load_state_dict(d)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="cardio vessel segmentation")
    parser.add_argument("--configs", dest="cfg",
                        help="The configs file.",
                        default='./configs/heart_server.yaml',
                        type=str)
    parser.add_argument('--iters', dest='iters', help='Iterations in training.', type=int, default=None)
    parser.add_argument('--batch_size', dest='batch_size', help='Mini batch size of one gpu or cpu.', type=int, default=None)
    parser.add_argument('--learning_rate', dest='learning_rate', help='Learning rate', type=float, default=None)
    parser.add_argument('--seed', dest='seed', help='seed', type=float, default=0)
    parser.add_argument('--mode', dest='mode', help='train, infer or both', type=str, default=None)

    parser.add_argument('--experiments_path', dest='experiments_path',
                        help='experiments_path, all output and checkpoint are saved here.', type=str, default=None)

    parser.add_argument('--pretrain_weight_path', help='pretrain weight for training, testing of inferring', type=str, default=None)
    parser.add_argument('--img_path', help='image path', type=str, default=None)

    parser.add_argument('--label_path', help='label path', type=str, default=None)
    parser.add_argument('--output_path', help='save path', type=str, default=None)
    parser.add_argument('--persist_path', help='persist path, the path used to save persist data in persist loader',
                        type=str, default=None)
    parser.add_argument('--val_set', help='val set, a txt path to select val data, only used if mode is train',
                        type=str, default=None)
    parser.add_argument('--train_set', help='train set, a txt path to select train data, only used if mode is train',
                        type=str, default=None)
    parser.add_argument('--dataset_information', help='dataset information',
                        type=str, default=None)

    # -------------------------------------------- multi information -------------------------------------------- #
    parser.add_argument('--CS_W',
                        help='pretrain weight from coarse segmentation, testing of inferring', type=str, default=None)
    parser.add_argument('--CS_M', help='coarse segmentation in main phase', type=str, default=None)
    parser.add_argument('--CS_A', help='coarse segmentation in auxiliary phase', type=str, default=None)
    parser.add_argument('--CS_DL', help='discontinuity label generated in main coarse segmentation', type=str,
                        default=None)
    parser.add_argument('--CS_DLGT', help='discontinuity label generated in auxiliary coarse segmentation', type=str,
                        default=None)
    parser.add_argument('--I_M', help='main image', type=str, default=None)
    parser.add_argument('--I_A', help='auxiliary image', type=str, default=None)
    parser.add_argument('--select_file', help='use discontinuity label to determine what data are used', type=str,
                        default=None)
    parser.add_argument('--delete_persist', help='pretrain weight from coarse segmentation, testing of inferring', type=str,)
    parser.add_argument('--view', action='store_true', default=False, help='delete persist cache, because it is too large')
    # -------------------------------------------- multi information -------------------------------------------- #

    args = parser.parse_args()
    monai.config.print_config()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size,
        seed=args.seed,
        mode=args.mode,
        img_path=args.img_path,
        I_M=args.I_M,
        I_A=args.I_A,
        CS_M=args.CS_M,
        CS_A=args.CS_A,
        CS_DL=args.CS_DL,
        CS_DLGT=args.CS_DLGT,
        CS_W=args.CS_W,
        select_file=args.select_file,
        label_path=args.label_path,
        output_path=args.output_path,
        persist_path=args.persist_path,
        val_set=args.val_set,
        train_set=args.train_set,
        experiments_path=args.experiments_path,
        pretrain_weight_path=args.pretrain_weight_path,
        dataset_information=args.dataset_information
        )
    set_determinism(seed=cfg.seed)

    if cfg.dic['mode'] == 'train':

        # ---------------------------- built transform sequence ---------------------------- #
        # ------------- take a look at val_transform on dataset and save a case ------------ #
        cfg.creat_training_require()
        load_weight_from_coarse_segmentation(in_channels=cfg.dic['model'].get("in_channels"), model=cfg.model,
                                             net_architecture=cfg.dic['model']['name'], weight_path=cfg.CS_W)
        train_transforms, val_transforms, save_transform = get_second_stage_only_one_phase(cfg.dic['transform'])
        if cfg.dic["train"]["loader"].get("file_path"):
            files = [load_json(cfg.dic["train"]["loader"]["file_path"])]
        elif cfg.dic["train"]["loader"].get("val_set"):
            val_files = prepare_main_with_img_datalist_with_file(main_file=cfg.CS_M,
                                                                 main_img_file=cfg.I_M,
                                                                 label_file=cfg.train_label_path,
                                                                 broken_file=cfg.CS_DL,
                                                                 broken_gt_file=cfg.CS_DLGT,
                                                                 img_name=cfg.val_set,
                                                                 select_file=cfg.select_file, )

            train_files = prepare_main_with_img_datalist_with_file(main_file=cfg.CS_M,
                                                                   main_img_file=cfg.I_M,
                                                                   label_file=cfg.train_label_path,
                                                                   broken_file=cfg.CS_DL,
                                                                   broken_gt_file=cfg.CS_DLGT,
                                                                   img_name=cfg.train_set,
                                                                   select_file=cfg.select_file, )
            files = [{"train_files": train_files, "val_files": val_files}]

        else:
            raise ValueError("please provide a file path or val set and train set")
        # check_transform_in_dataloader(val_files=files[0]['val_files'], val_transforms=try_transforms)

        # ---------------------------- create loss function ---------------------------- #
        # ------------- you can crate your own loss or metric here amd replace cfg ------------ #

        # dice_metric = DiceMetric(include_background=False, reduction="mean")
        # loss_function = DiceLoss(to_onehot_y=True, softmax=True)

        # -------------- training ---------------- #
        metric_record = []
        experiment_path = os.path.join(cfg.dic['experiments_path'], time.strftime("%d_%m_%Y_%H_%M_%S"))\
            if cfg.dic.get('time_name', None) else cfg.dic['experiments_path']

        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        for i in range(len(files)):
            if cfg.dic['train']['loader'].get('split_mode') == "five_fold":
                experiment_path_fold = os.path.join(experiment_path, f"{i}_fold")
                if not os.path.exists(experiment_path_fold):
                    os.makedirs(experiment_path_fold)
            else:
                experiment_path_fold = experiment_path
            write_data_reference(files[i], experiment_path_fold)
            save_json(files[i], os.path.join(experiment_path_fold, 'files.txt'))
            # -------------- Create Model, Loss, Optimizer in Config-------------- #
            # ------------------------ using config ------------------------------ #
            cfg.save_config(os.path.join(experiment_path_fold, 'configs.yaml'))
            if cfg.dic['train']['loader'].get('split_mode') == "five_fold":
                print(f"----------------fold{i} start!-----------------")
            else:
                print(f"----------------training start!-----------------")
            if cfg.dic['train']['loader'].get('persist'):
                if cfg.persist_path == 'default':
                    persistent_cache = pathlib.Path(experiment_path_fold, "persistent_cache")
                else:
                    persistent_cache = pathlib.Path(cfg.persist_path)

                persistent_cache.mkdir(parents=True, exist_ok=True)
                train_ds = PersistentDataset(data=files[i]['train_files'], transform=train_transforms,
                                             cache_dir=persistent_cache)
                val_ds = PersistentDataset(data=files[i]['val_files'], transform=val_transforms,
                                           cache_dir=persistent_cache)
            else:
                train_ds = CacheDataset(
                    data=files[i]['train_files'], transform=train_transforms,
                    cache_rate=cfg.dic['train']['loader']['cache'], num_workers=4)
                # train_ds = Dataset(data=train_files, transform=train_transforms)
                # use batch_size=2 to load images and use RandCropByPosNegLabeld
                val_ds = CacheDataset(
                    data=files[i]['val_files'], transform=val_transforms,
                    cache_rate=cfg.dic['train']['loader']['cache'], num_workers=4)

            # load pretrained model
            out_channels = cfg.dic['model']['out_channels'] if cfg.dic['model'].get('out_channels', None) else cfg.model.out_channels
            cardio_vessel_segmentation_multi_phase_with_image_train(cfg=cfg,
                                                                    key=cfg.second_stage_key,
                                                                    model=cfg.model,
                                                                    num_class=out_channels,
                                                                    loss_function=cfg.train_loss,
                                                                    val_metric=cfg.val_metric,
                                                                    optimizer=cfg.optimizer_init,
                                                                    lr_scheduler=cfg.lr_scheduler_init,
                                                                    train_dataset=train_ds,
                                                                    val_dataset=val_ds,
                                                                    experiment_path=experiment_path_fold,
                                                                    device=cfg.device,
                                                                    start_epoch=cfg.start_epoch,
                                                                    sw_batch_size=cfg.train_sw_batch_size,
                                                                    overlap=cfg.train_sw_overlap,
                                                                    mirror_axes=cfg.train_mirror_axes,
                                                                    )
            if cfg.dic['train']['loader'].get('persist'):
                import shutil
                shutil.rmtree(persistent_cache)

    elif cfg.dic['mode'] == 'test':
        train_transforms, val_transforms, save_transform = get_second_stage_only_one_phase(cfg.dic['transform'])

        cfg.creat_test_require()
        cfg.save_config(os.path.join(cfg.test_output_path, 'config.yaml'))
        if cfg.dic["test"]["loader"].get("file_path"):
            files = load_json(cfg.dic["test"]["loader"]["file_path"])["val_files"]
        elif cfg.dic["test"]["loader"].get("val_set"):
            files = prepare_main_with_img_datalist_with_file(main_file=cfg.CS_M,
                                                             main_img_file=cfg.I_M,
                                                             label_file=cfg.test_label_path,
                                                             broken_file=cfg.CS_DL,
                                                             broken_gt_file=cfg.CS_DLGT,
                                                             img_name=cfg.val_set,
                                                             select_file=cfg.select_file, )
        else:
            files = prepare_main_with_img_datalist_with_file(main_file=cfg.CS_M,
                                                             main_img_file=cfg.I_M,
                                                             label_file=cfg.test_label_path,
                                                             broken_file=cfg.CS_DL,
                                                             broken_gt_file=cfg.CS_DLGT,
                                                             img_name=None,
                                                             select_file=None, )
        # ---------------------------- built transform sequence ---------------------------- #
        total_ds = CacheDataset(
            data=files, transform=val_transforms, cache_rate=cfg.dic['test']['loader']['cache'], num_workers=2)

        # ------------------------------------- test --------------------------------------- #
        cardio_vessel_segmentation_multi_phase_with_image_test(model=cfg.model,
                                                               key=['I_M', 'CS_M', 'CS_A', 'CS_DLGT'],
                                                               val_dataset=total_ds,
                                                               device=cfg.device,
                                                               output_path=cfg.test_output_path,
                                                               window_size=cfg.dic['transform']['patch_size'],
                                                               save_data=cfg.dic['test']['save_data'])

    elif cfg.dic['mode'] == 'infer':
        infer_transforms = get_second_stage_only_one_phase(cfg.dic['transform'], mode='infer')

        cfg.creat_infer_require()
        cfg.save_config(os.path.join(cfg.infer_output_path, 'config.yaml'))

        # ---------------------------- built transform sequence ---------------------------- #
        # files = prepare_image_list(image_path=cfg.dic['infer']['loader']['path'])

        files = prepare_main_with_img_datalist_with_file(main_file=cfg.CS_M,
                                                         main_img_file=cfg.I_M,
                                                         label_file=None,
                                                         broken_file=cfg.CS_DL,
                                                         broken_gt_file=cfg.CS_DLGT,
                                                         img_name=None,
                                                         select_file=None, )
        total_ds = CacheDataset(
            data=files, transform=infer_transforms, cache_rate=cfg.dic['infer']['loader']['cache'], num_workers=2)

        cardio_vessel_segmentation_multi_phase_with_image_infer(model=cfg.model,
                                                                key=cfg.second_stage_key,
                                                                val_dataset=total_ds,
                                                                device=cfg.device,
                                                                output_path=cfg.infer_output_path,
                                                                window_size=tuple(cfg.dic['transform']['patch_size']),
                                                                origin_transforms=infer_transforms,
                                                                overlap=cfg.infer_sw_overlap,
                                                                sw_batch_size=cfg.infer_sw_batch_size,
                                                                mirror_axes=cfg.infer_mirror_axes
                                                                )
    else:
        raise RuntimeError('Only train and infer mode are supported now')


