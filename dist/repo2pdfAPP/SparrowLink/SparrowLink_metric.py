from post_processing.fracture_detection import sparrowlink_metric
import numpy as np
import SimpleITK as sitk
import json
from openpyxl import Workbook

class SparrowLinkMetric(object):
    def __init__(self,
                 max_broken_size_percentage=0.1,
                 min_sphere_radius_percentage=0.02,
                 sphere_dilation_percentage=0.01,
                 region_select=4,
                 cube_min_size=(10, 10, 10),
                 skeleton_refine_times=3,
                 region_threshold=5,
                 angle_threshold_1=0.0,
                 angle_threshold_2=0.0,
                 view=False
                 ):
        self.max_broken_size_percentage = max_broken_size_percentage
        self.min_sphere_radius_percentage = min_sphere_radius_percentage
        self.sphere_dilation_percentage = sphere_dilation_percentage
        self.region_select = region_select
        self.cube_min_size = cube_min_size
        self.skeleton_refine_times = skeleton_refine_times
        self.region_threshold = region_threshold
        self.angle_threshold_1 = angle_threshold_1
        self.angle_threshold_2 = angle_threshold_2
        self.view = view
    """
    :param max_broken_size_percentage: the max broken size percentage, used in constriction on distance
    :param min_sphere_radius_percentage: the min sphere radius percentage, used for some small fracture
    :param sphere_dilation_percentage: the sphere dilation percentage, to consider the structure around fracture area
    :param region_select: the region select, matching points consist of start point and paired point, start point is...
    ...constricted in the region with region_select.
    :param cube_min_size: the min size of the cube, used for small fracture
    :param skeleton_refine_times: the refine times of the skeleton, zhang algorithm might cause some small branch in the...
    ...skeleton, so we need to refine the skeleton to delete the small branch
    :param region_threshold: the region threshold, used to delete the small region in the segmentation
    :param angle_threshold_1: the angle threshold 1, used for constriction on orientation
    :param angle_threshold_2: the angle threshold 2, used for constriction on orientation
    """
    def __call__(self,
                 seg_path: str = None,
                 gt_path: str = None,
                 save_path: str = None
                 ):

        seg, spacing, origin, direction = self.read_image(seg_path)
        self.spacing, self.origin, self.direction = spacing, origin, direction
        gt, _, _, _ = self.read_image(gt_path)
        result_dict = sparrowlink_metric(
            label=seg,
            GT=gt,
            spacing=spacing,
            direction=direction,
            origin=origin,
            save_path=save_path,
            view=self.view,
            max_broken_size_percentage=self.max_broken_size_percentage,
            min_sphere_radius_percentage=self.min_sphere_radius_percentage,
            sphere_dilation_percentage=self.sphere_dilation_percentage,
            cube_min_size=self.cube_min_size,
            skeleton_refine_times=self.skeleton_refine_times,
            region_threshold=self.region_threshold,
            angle_threshold_1=self.angle_threshold_1,
            angle_threshold_2=self.angle_threshold_2,
        )
        d = {
            "name": pathlib.Path(seg_path).name[:9],
            "num_gt": result_dict["num_gt"],
        }
        ##### permenant #####
        if "_CS_M" in str(seg_path):
            print(1)
            sphere_gt = result_dict["mask_sphere_gt"]
            gt_segment = ((sphere_gt * gt) > 0).astype(np.uint16)
            self.save_image(gt_segment, save_path=seg_path.replace(".nii.gz", "_sphere_gt_segment.nii.gz"))
            self.save_image(sphere_gt, save_path=seg_path.replace(".nii.gz", "_sphere_gt.nii.gz"))
            path = seg_path.replace("_CS_M.nii.gz", "_RCS_NEW.nii.gz")
            rcs_new = self.read_image(path)[0]
            rcs_new_gt_sphere_segment = ((sphere_gt * rcs_new) > 0).astype(np.uint16)
            self.save_image(rcs_new_gt_sphere_segment,
                            save_path=seg_path.replace(".nii.gz", "_rcs_new_gt_sphere_segment.nii.gz"))
            cs_m_gt_sphere_segment = ((sphere_gt * seg) > 0).astype(np.uint16)
            self.save_image(cs_m_gt_sphere_segment,
                            save_path=seg_path.replace(".nii.gz", "_CS_M_gt_sphere_segment.nii.gz"))
        return d

    def save_image(self, image, save_path, save_type=np.uint16):
        sitk_image = sitk.GetImageFromArray(image.astype(save_type))
        sitk_image.SetSpacing(self.spacing)
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetDirection(self.direction)
        sitk.WriteImage(sitk_image, save_path)

    @staticmethod
    def read_image(path):
        sitk_image = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(sitk_image)
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        return image, spacing, origin, direction


def update(pbar, record, result):
    pbar.update()
    record.append(result)


def error_back(err):
    print(err)


if __name__ == "__main__":
    import argparse
    import tqdm
    import pathlib
    from multiprocessing import Pool

    import warnings
    warnings.filterwarnings("ignore")  # ignore from np.int16 to np.uint8

    parser = argparse.ArgumentParser()
    parser.add_argument("--seg", type=str, default=None)
    parser.add_argument('--seg_find', type=str, default="*.nii.gz")
    parser.add_argument("--gt", type=str, default=None)
    parser.add_argument('--gt_find', type=str, default="*.nii.gz")
    parser.add_argument("--multiprocess", action='store_true', default=False)
    parser.add_argument('--metric_postfix', type=str, default=None)
    args = parser.parse_args()

    assert args.seg is not None, "seg_path is None"
    assert args.gt is not None, "label_path is None"
    if args.metric_postfix is None:
        metric_result_path = str(pathlib.Path(args.seg) / "sparrowlink_metric.xlsx")
    else:
        metric_result_path = str(pathlib.Path(args.seg) / f"sparrowlink_metric_{args.metric_postfix}.xlsx")

    wb = Workbook()
    ws = wb.active
    ws.append(['name', 'num_gt'])
    metric_list = []

    seg = pathlib.Path(args.seg)
    gt = pathlib.Path(args.gt)
    seg_list = list(seg.glob(f"{args.seg_find}"))
    gt_list = list(gt.glob(f"{args.gt_find}"))
    seg_list.sort()
    gt_list.sort()
    print(f"\033[96m SparrowLink Metric Calculating for {args.seg_find} \033[00m")
    pbar = tqdm.tqdm(total=len(seg_list), colour="#87cefa")
    pbar.set_description("SparrowLink Processing")
    metric_record = []
    poor = Pool(14)
    for seg_path, gt_path in zip(seg_list, gt_list):
        name = seg_path.name[:9]
        metric_calculator = SparrowLinkMetric(view=False)
        if not args.multiprocess:
            result_dict = metric_calculator(
                seg_path=str(seg_path),
                gt_path=str(gt_path),
                save_path=None
            )
            metric_record.append(result_dict)
            pbar.update()
        else:
            kwargs = {
                "seg_path": str(seg_path),
                "gt_path": str(gt_path),
                "save_path": None
            }
            poor.apply_async(
                func=metric_calculator,
                kwds=kwargs,
                callback=lambda x: update(pbar, metric_record, x),
                error_callback=error_back
            )
    metric_record.sort(key=lambda x: x.get("name"))
    if args.multiprocess:
        poor.close()
        poor.join()

    for result in metric_record:
        ws.append([result["name"], result["num_gt"]])

    mean_metric = np.array([metric_record[i].get('num_gt') for i in range(len(metric_record))]).mean(axis=0).tolist()
    ws.append(["mean", mean_metric])
    wb.save(metric_result_path)










