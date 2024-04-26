import pathlib
import ants
import time
import argparse
from multiprocessing import Pool


def Reg(fixed_path, moving_path, apply_path, save_path, save_path_apply, mode='SyNRA', reg_interpolator='linear',
        apply_interpolator='nearestNeighbor'):
    tic = time.time()
    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))
    apply = ants.image_read(str(apply_path))
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=mode)
    moving_reg = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=reg['fwdtransforms'],
                                       interpolator=reg_interpolator)
    apply_reg = ants.apply_transforms(fixed=fixed, moving=apply, transformlist=reg['fwdtransforms'],
                                      interpolator=apply_interpolator)

    ants.image_write(moving_reg, str(save_path / moving_path.name))
    ants.image_write(apply_reg, str(save_path_apply / moving_path.name))

    return {"name": moving_path.name, "time": time.time()-tic}


def update(pbar, result):
    pbar.update()
    # print(result)


def call_fun(result):
    print(result)


def errorback(err):
    print(err)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--reg_algorithm', type=str, default='SyNRA')  # SyNRA, Rigid
    arg.add_argument('--fixed_dir', type=str,
                     default="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/data/ZhangX/CT_Img/")
    arg.add_argument('--moving_dir', type=str,
                     default="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/data/ZhangX/CTA_Img/")
    arg.add_argument('--apply_dir', type=str,
                     default="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/data/ZhangX/CTA_infer/")
    arg.add_argument('--save_dir', type=str,
                     default="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/data/ZhangX/CTA_Img_reg/")
    arg.add_argument('--save_dir_apply', type=str,
                     default="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/data/ZhangX/CTA_infer_reg/")


    args = arg.parse_args()
    fixed_dir = pathlib.Path(args.fixed_dir)
    moving_dir = pathlib.Path(args.moving_dir)
    apply_dir = pathlib.Path(args.apply_dir)
    save_dir = pathlib.Path(args.save_dir)
    save_dir_apply = pathlib.Path(args.save_dir_apply)
    save_dir.mkdir(exist_ok=True)
    save_dir_apply.mkdir(exist_ok=True)
    print(f"---------{args.reg_algorithm}_registration---------")
    pool = Pool(4)
    for file in fixed_dir.glob('*.nii.gz'):
        name = file.name
        moving_path = moving_dir / name.replace('CT', 'CTA')
        apply_path = apply_dir / name.replace('CT', 'CTA')
        pool.apply_async(Reg,
                         args=(file, moving_path, apply_path, save_dir, save_dir_apply, args.reg_algorithm),
                         callback=call_fun,
                         error_callback=errorback)

    pool.close()
    pool.join()
