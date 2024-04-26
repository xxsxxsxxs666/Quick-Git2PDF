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
from data.loader import prepare_datalist, prepare_datalist_with_file, prepare_image_list, save_json, write_data_reference, load_json
from transform.utils import get_transform
from utils.test import cardio_vessel_segmentation_test
from utils.trainer import cardio_vessel_segmentation_train
from utils.inferer import cardio_vessel_segmentation_infer
from pre_processing.checking import DatasetinformationExtractor
import pathlib
# print_config()


torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument("--configs",
                        dest="cfg",
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
        label_path=args.label_path,
        output_path=args.output_path,
        persist_path=args.persist_path,
        val_set=args.val_set,
        train_set=args.train_set,
        experiments_path=args.experiments_path,
        pretrain_weight_path=args.pretrain_weight_path,
        )
    set_determinism(seed=cfg.seed)
    if cfg.dic['mode'] == 'train':

        # ---------------------------- built transform sequence ---------------------------- #
        # ---------------------- Create Model, Loss, Optimizer in Config ------------------- #
        cfg.creat_training_require()
        train_transforms, val_transforms, save_transform = get_transform(cfg.dic['transform'])
        if cfg.dic["train"]["loader"].get("file_path"):
            files = [load_json(cfg.dic["train"]["loader"]["file_path"])]
        elif cfg.dic["train"]["loader"].get("val_set"):
            val_files = prepare_datalist_with_file(image_file=cfg.train_img_path,
                                                   label_file=cfg.train_label_path,
                                                   img_name=cfg.val_set, )
            train_files = prepare_datalist_with_file(image_file=cfg.train_img_path,
                                                     label_file=cfg.train_label_path,
                                                     img_name=cfg.train_set, )
            files = [{"train_files": train_files, "val_files": val_files}]
        else:
            files = prepare_datalist(image_file=cfg.train_img_path,
                                     label_file=cfg.train_label_path,
                                     split_mode=cfg.dic['train']['loader']['split_mode'], )
        # ---------------------------- create loss function ----------------------------------- #
        # ------------- you can crate your own loss or metric here amd replace cfg ------------ #

        # dice_metric = DiceMetric(include_background=False, reduction="mean")
        # loss_function = DiceLoss(to_onehot_y=True, softmax=True)

        # ---------------------------------- training ----------------------------------------- #
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
            # --------------------------------- save config ------------------------------ #
            cfg.save_config(os.path.join(experiment_path_fold, 'configs.yaml'))
            if cfg.dic['train']['loader'].get('split_mode') == "five_fold":
                print(f"----------------fold{i} start!-----------------")
            else:
                print(f"----------------training start!-----------------")
            if cfg.dic['train']['loader'].get('persist'):
                print("---------------- using persist dataset -----------------")
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
                print("---------------- using cache dataset -----------------")
                train_ds = CacheDataset(
                    data=files[i]['train_files'], transform=train_transforms,
                    cache_rate=cfg.dic['train']['loader']['cache'], num_workers=cfg.dic['train']['loader']['num_workers'])
                # train_ds = Dataset(data=train_files, transform=train_transforms)
                # use batch_size=2 to load images and use RandCropByPosNegLabeld
                val_ds = CacheDataset(
                    data=files[i]['val_files'], transform=val_transforms,
                    cache_rate=cfg.dic['train']['loader']['cache'], num_workers=cfg.dic['train']['loader']['num_workers'])
            out_channels = cfg.dic['model']['out_channels'] if cfg.dic['model'].get('out_channels',
                                                                                    None) else cfg.model.out_channels
            cardio_vessel_segmentation_train(cfg=cfg,
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
                                             metric_record=metric_record,
                                             start_epoch=cfg.start_epoch,
                                             mirror_axes=cfg.train_mirror_axes,
                                             sw_batch_size=cfg.train_sw_batch_size,
                                             overlap=cfg.train_sw_overlap,
                                             )

    elif cfg.dic['mode'] == 'test':
        # ---------------------------- built transform sequence ---------------------------- #
        train_transforms, val_transforms, save_transform = get_transform(cfg.dic['transform'])

        cfg.creat_test_require()
        cfg.save_config(os.path.join(cfg.test_output_path, 'config.yaml'))
        if cfg.dic["test"]["loader"].get("file_path"):
            files = load_json(cfg.dic["test"]["loader"]["file_path"])["val_files"]
        elif cfg.dic["test"]["loader"].get("val_set"):
            val_files = prepare_datalist_with_file(image_file=cfg.test_img_path,
                                                   label_file=cfg.test_label_path,
                                                   img_name=cfg.dic["test"]["loader"]["val_set"], )
            files = val_files
        else:
            files = prepare_datalist(image_file=cfg.train_img_path,
                                     label_file=cfg.test_label_path,
                                     split_mode=cfg.dic['test']['loader']['split_mode'], )

        total_ds = CacheDataset(
            data=files, transform=val_transforms, cache_rate=cfg.dic['test']['loader']['cache'], num_workers=2)

        cardio_vessel_segmentation_test(model=cfg.model,
                                        val_dataset=total_ds,
                                        device=cfg.device,
                                        output_path=cfg.test_output_path,
                                        window_size=cfg.dic['test']['test_windows_size'],
                                        save_data=cfg.dic['test']['save_data'])

    elif cfg.dic['mode'] == 'infer':
        # ---------------------------- built transform sequence ---------------------------- #
        infer_transforms = get_transform(cfg.dic['transform'], mode='infer')

        cfg.creat_infer_require()
        cfg.save_config(os.path.join(cfg.infer_output_path, 'config.yaml'))

        # files = prepare_image_list(image_path=cfg.dic['infer']['loader']['path'])

        image_path = pathlib.Path(cfg.infer_img_path)
        assert image_path.is_dir(), f"img path not exist: {image_path}"
        # get the image name list in image_path using pathlib
        train_images = [path.name for path in image_path.glob("*.nii.gz")]
        data_dicts = [
            {"image": str(pathlib.Path(image_path, image_name))}
            for image_name in train_images
        ]
        total_ds = CacheDataset(
            data=data_dicts, transform=infer_transforms, cache_rate=cfg.dic['infer']['loader']['cache'], num_workers=2)
        cardio_vessel_segmentation_infer(model=cfg.model,
                                         val_dataset=total_ds,
                                         device=cfg.device,
                                         output_path=cfg.infer_output_path,
                                         window_size=tuple(cfg.dic['transform']['patch_size']),
                                         overlap=cfg.infer_sw_overlap,
                                         origin_transforms=infer_transforms,
                                         mirror_axes=cfg.infer_mirror_axes,
                                         sw_batch_size=cfg.infer_sw_batch_size,
                                         mode="gaussian",
                                         )
    else:
        raise RuntimeError('Only train and infer mode are supported now')


