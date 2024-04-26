from monai.transforms import (
    LoadImage,
    SaveImage,
)
from tqdm import tqdm
from monai.data.meta_tensor import MetaTensor
import monai
import numpy as np
from openpyxl import Workbook
from metric_zoo import clDice
import seg_metrics.seg_metrics as sg
import argparse
import pathlib

dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean")  # only label channel
hausdorff_metric = monai.metrics.HausdorffDistanceMetric(include_background=True, reduction="mean")
asd_metric = monai.metrics.SurfaceDistanceMetric(include_background=True, symmetric=True)
iou_metric = monai.metrics.MeanIoU(include_background=True, reduction="mean")


def update(pbar, record, result):
    pbar.update()
    record.append(result)


def error_back(err):
    print(err)


def caculate_metric(label_root, seg_root):
    label = LoadImage(image_only=False)(str(label_root))[0]
    seg = LoadImage(image_only=False)(str(seg_root))[0]
    if label.sum() == 0:
        result = {"name": [label_root.name], "metric": [1000, 1000, 1000,
                                                        1000, 1000]}
        return result
    if seg.sum() == 0:
        result = {"name": [label_root.name], "metric": [0, 0, 0,
                                                        1000, 1000]}
        return result
    dice = dice_metric(y_pred=seg[None, None], y=label[None, None])
    cldice = clDice(seg, label)
    iou = iou_metric(y_pred=seg[None, None], y=label[None, None])
    spacing = label.meta["pixdim"][1:4].tolist()
    distance_metrics = sg.write_metrics(labels=[1],  # exclude background if needed
                                        gdth_img=np.array(label),
                                        pred_img=np.array(seg),
                                        csv_file=None,
                                        spacing=spacing,
                                        metrics=['msd', 'hd95'],
                                        verbose=False)
    result = {"name": [label_root.name], "metric":[dice.item(), cldice.item(), iou.item(),
                                                   distance_metrics[0]['msd'][0],distance_metrics[0]['hd95'][0]]}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', type=str, default=None)
    parser.add_argument('--label_path', type=str, default=None)
    parser.add_argument('--metric_result_path', type=str, default=None)
    parser.add_argument('--label_find', type=str, default="*.nii.gz")
    parser.add_argument('--seg_find', type=str, default="*.nii.gz")
    parser.add_argument("--multiprocess", action='store_true', default=False)
    args = parser.parse_args()
    assert args.seg_path is not None, "seg_path is None"
    assert args.label_path is not None, "label_path is None"
    if args.metric_result_path is None:
        metric_result_path = str(pathlib.Path(args.seg_path) / "metric_result.xlsx")
    else:
        metric_result_path = args.metric_result_path
        pathlib.Path(metric_result_path).parent.mkdir(parents=True, exist_ok=True)
    # caculate the dice metric of the segmentation result

    label_list = list(pathlib.Path(args.label_path).glob(args.label_find))
    seg_list = list(pathlib.Path(args.seg_path).glob(args.seg_find))
    # seg_list = [name for name in seg_list if 'auxiliary' not in name]
    label_list.sort()
    seg_list.sort()
    # save the image_name, dice metric in a excel file
    # new a excel file
    wb = Workbook()
    ws = wb.active
    ws.append(['image_name', 'dice_metric', "cldice", "mIoU", "MSD", "HD95"])
    metric_list = []

    print(f"\033[96m calculating metric for {args.seg_find} \033[00m")
    assert len(seg_list) > 0, f"no file found in {args.seg_path} {args.seg_find}"
    pbar = tqdm(total=len(label_list), colour='#87cefa')
    if not args.multiprocess:
        for label_root, seg_root in zip(label_list, seg_list):
            # print(label_root.split("/")[-1], seg_root.split("/")[-1])
            result_dict = caculate_metric(label_root, seg_root)
            ws.append(result_dict["name"] + result_dict["metric"])
            metric_list.append(result_dict["metric"])
            pbar.set_description(f'metric:{str(seg_root.name), result_dict["metric"][0]}')
            pbar.update()
    else:
        from multiprocessing import Pool
        pool = Pool(14)
        result_record = []
        for label_root, seg_root in zip(label_list, seg_list):
            pool.apply_async(func=caculate_metric,
                             args=(label_root, seg_root),
                             error_callback=error_back,
                             callback=lambda x: update(pbar, result_record, x))
        pool.close()
        pool.join()
        # sort the result by "name"
        result_record.sort(key=lambda x: x.get("name"))
        for result in result_record:
            ws.append(result.get("name") + result.get("metric"))
            metric_list.append(result.get("metric"))

    metric_array = np.array(metric_list)
    mean_metric = metric_array.mean(axis=0).tolist()
    mean_metric.insert(0, 'mean')
    ws.append(mean_metric)
    wb.save(metric_result_path)
