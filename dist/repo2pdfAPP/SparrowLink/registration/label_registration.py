import pathlib
import ants
import time
import argparse
from multiprocessing import Pool


def Reg(target_path, moving_path, target_path_1, moving_path_1, save_path, save_path_1, mode='SyNRA', t=1):
    tic = time.time()
    target = ants.image_read(str(target_path))
    moving = ants.image_read(str(moving_path))
    target_1 = ants.image_read(str(target_path_1))
    moving_1 = ants.image_read(str(moving_path_1))
    moving_loop = moving
    moving_1_loop = moving_1
    target_loop = target
    target_1_loop = target_1
    for i in range(t):
        reg = ants.registration(fixed=target, moving=moving_loop, type_of_transform=mode)
        moving_loop = ants.apply_transforms(fixed=target, moving=moving_loop, transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
        moving_1_loop = ants.apply_transforms(fixed=target_1, moving=moving_1_loop, transformlist=reg['fwdtransforms'])
        target_loop = ants.apply_transforms(fixed=moving, moving=target_loop, transformlist=reg['invtransforms'], interpolator='nearestNeighbor')
        target_1_loop = ants.apply_transforms(fixed=moving_1_loop, moving=target_1_loop, transformlist=reg['invtransforms'])


    ants.image_write(moving_loop, str(save_path))
    ants.image_write(moving_1_loop, str(save_path_1))

    save_path_2 = save_path.parent.parent / save_path.parent.name.replace('auxiliary', 'main')
    save_path_2.mkdir(exist_ok=True, parents=True) if not save_path_2.exists() else None
    save_path_2 = save_path_2 / save_path.name
    save_path_3 = save_path_1.parent.parent / save_path_1.parent.name.replace('auxiliary', 'main')
    save_path_3.mkdir(exist_ok=True, parents=True)
    save_path_3 = save_path_3 / save_path_1.name

    ants.image_write(target_loop, str(save_path_2))
    ants.image_write(target_1_loop, str(save_path_3))

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
    arg.add_argument('--mode', type=str, help='used in save name', default='test')
    arg.add_argument('--time', type=int, help='time for registration loop',default=1)
    arg.add_argument('--net', type=str, default="used in save name")
    arg.add_argument('--save_root', type=str, help='save path for img and mask after registration',
                     default="/public/home/v-xiongxx/Graduate_project/"
                             "Cardio_vessel_segmentaion_based_on_monai/"
                             "experiments/Graduate_project/multi_phase/pretrain")
    arg.add_argument('--main_mask_path', type=str, default=None)
    arg.add_argument('--auxiliary_mask_path', type=str, default=None)
    arg.add_argument('--main_img_path', type=str, default=None)
    arg.add_argument('--auxiliary_img_path', type=str, default=None)


    args = arg.parse_args()
    main_label_path = pathlib.Path(args.main_mask_path)
    auxiliary_label_path = pathlib.Path(args.auxiliary_mask_path)
    main_img_path = pathlib.Path(args.main_img_path)
    auxiliary_img_path = pathlib.Path(args.auxiliary_img_path)
    save_auxiliary_label_regitration = pathlib.Path(args.save_root) / f"auxiliary_{args.mode}_infer_{args.reg_algorithm}_reg_{args.time}"
    save_auxiliary_img_regitration = pathlib.Path(args.save_root) / f"auxiliary_{args.mode}_img_{args.reg_algorithm}_reg_{args.time}"
    save_auxiliary_img_regitration.mkdir(exist_ok=True, parents=True)
    save_auxiliary_label_regitration.mkdir(exist_ok=True, parents=True)
    # find the corresponding file in segment_syst
    main_list = list(main_img_path.glob('*.nii.gz'))
    # sort the file name
    main_list.sort(key=lambda x: str(x.stem))
    # q: what is x.stem?

    pool = Pool(4)
    print(f"---------{args.mode}_registration---------")
    for i, img in enumerate(main_list):
        img_name = img.name

        main_img = img
        main_label = main_label_path / img_name
        auxiliary_img = auxiliary_img_path / img_name
        auxiliary_label = auxiliary_label_path / img_name
        save_auxiliary_label = save_auxiliary_label_regitration / img_name
        save_auxiliary_img = save_auxiliary_img_regitration / img_name

        assert main_label.exists(), f"{str(main_label)} not exist"
        assert main_img.exists(), f"{str(main_img)} not exist"
        assert auxiliary_img.exists(), f"{str(auxiliary_img)} not exist"
        assert auxiliary_label.exists(), f"{str(auxiliary_label)} not exist"
        # Reg(target_path=main_label,
        #     moving_path=auxiliary_label,
        #     target_path_1=main_img,
        #     moving_path_1=auxiliary_img,
        #     save_path=save_auxiliary_img,
        #     save_path_1=save_auxiliary_label,
        #     mode=args.mode)
        pool.apply_async(Reg,
                         args=(main_label, auxiliary_label, main_img, auxiliary_img, save_auxiliary_label, save_auxiliary_img, args.reg_algorithm, args.time),
                         callback=call_fun,
                         error_callback=errorback)

    pool.close()
    pool.join()
