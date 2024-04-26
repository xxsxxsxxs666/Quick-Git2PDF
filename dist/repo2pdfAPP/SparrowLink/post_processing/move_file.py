import pathlib
import argparse
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        help='move discontinuity detection file to sphere or cube file for easiler dataloading',
                        default='test')
    parser.add_argument('--save_path',
                        type=str,
                        help='save path',
                        default='test')
    parser.add_argument('--data_postfix',
                        type=str,
                        help='sphere or cube',
                        default='')
    parser.add_argument('--hierarchical',
                        type=str,
                        help='sphere or cube',
                        default='')
    parser.add_argument('--save_postfix',
                        type=str,
                        help='discontinuity detection file',
                        default='')
    parser.add_argument('--separate_folder',
                        action="store_true",
                        default=False,
                        )

    args = parser.parse_args()
    data_path = pathlib.Path(args.data_path)
    save_path = pathlib.Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    data_list = list(data_path.glob(f'*{args.hierarchical}{args.data_postfix}.nii.gz'))
    assert len(data_list) > 0, f"no file found in {data_path} {args.data_postfix}"
    print(f"move file for better viewing from \n"
          f"{args.data_path} to \n"
          f"{args.save_path}, \n"
          f"finding={args.hierarchical}{args.data_postfix}, save_postfix={args.save_postfix}",)
    pbar = tqdm(total=len(data_list))
    pbar.set_description('copy file')
    for file in data_list:
        file_name = file.name
        new_file_name = file_name.replace(f'{args.data_postfix}.nii.gz', f'{args.save_postfix}.nii.gz')
        if args.separate_folder:
            patient_id = file_name[:9]
            (save_path / patient_id).mkdir(exist_ok=True, parents=True)
            new_file = save_path / patient_id / new_file_name
        else:
            new_file = save_path / new_file_name
        shutil.copy(file, new_file)
        pbar.update()

