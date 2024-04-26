from post_processing.fracture_detection import run as discontinuity_detection
from post_processing.merge import merge
from post_processing.select_two_region import select_two_biggest_connected_region
import numpy as np
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import SimpleITK as sitk
import pathlib
import cc3d
import shutil
import json


class SparrowLinkPostProcess:
    def __init__(self,
                 cs_m_path: str = None,
                 cs_a_path: str = None,
                 rs_path: str = None,
                 cs_dl_path: str = None,
                 save_path: str = None,
                 max_broken_size_percentage=0.1,
                 ):
        """
        cs_m_path: path to the cs_m (coarse segmentation in main phase)
        cs_a_path: path to the cs_a (coarse segmentation in auxiliary phase)
        rs_a_path: path to the rs_a (refined segmentation in auxiliary phase)
        cs_dl_path: path to the cs_dl (coarse segmentation discontinuity label)
        """
        assert cs_m_path is not None, "cs_m_path is None"
        self.cs_m_path = cs_m_path
        self.cs_a_path = cs_a_path
        self.rs_path = rs_path
        self.cs_dl_path = cs_dl_path
        self.save_path = save_path
        self.name = pathlib.Path(cs_m_path).name[:9] # need to be modify to fit the other data
        self.save_subdir = pathlib.Path(save_path) / self.name
        self.save_subdir.mkdir(parents=True, exist_ok=True)
        self.discontinuity_detection_default = SparrowLinkDiscontinuityDetection(view=False,
                                                                                 max_broken_size_percentage=max_broken_size_percentage)
        self.discontinuity_detection = SparrowLinkDiscontinuityDetection(angle_threshold_2=-1,
                                                                         region_select=None,
                                                                         view=False,
                                                                         max_broken_size_percentage=max_broken_size_percentage)
        self.spacing = None
        self.origin = None
        self.direction = None

    def run(self):
        """
        Run the post processing pipeline.
        1. load the data
        2. directly merge rs and cs_m with cs_dl -> rcs
        3. select_two_biggest_connected_region on rcs -> rcs_two
        4. run discontinuity detection on rcs with cs_a and cs_dl -> arcs
        5. select_two_biggest_connected_region on arcs -> arcs_two

        if selected merge:
        6. selectively merge rs and cs_m with cs_dl -> rcs_selected
        7. select_two_biggest_connected_region on rcs_select -> rcs_selected_two
        8. run final merge on rcs_select with cs_a and cs_dl -> arcs_selected
        9. select_two_biggest_connected_region on arcs_selected -> arcs_selected_two

        if new merge:
        10. selectively merge rs and cs_m without cs_dl -> rcs_new
        11. select_two_biggest_connected_region on rcs_new -> rcs_new_two
        12. run final merge on rcs_new with cs_a and cs_dl -> arcs_new
        13. select_two_biggest_connected_region on arcs_new -> arcs_new_two
        """
        cs_a, rs, cs_dl, cs_m = None, None, None, None
        # 1. load the data
        cs_m, spacing, origin, direction = self.read_image(self.cs_m_path)
        self.spacing, self.origin, self.direction = spacing, origin, direction

        cs_a, _, _, _ = self.read_image(self.cs_a_path)
        rs, _, _, _ = self.read_image(self.rs_path)
        cs_dl, _, _, _ = self.read_image(self.cs_dl_path)

        # 2. directly merge rs and cs_m with cs_dl -> rcs
        rcs = self.merge(m1=cs_m, m2=rs, dl=cs_dl)

        # 3. select_two_biggest_connected_region on rcs -> rcs_two
        rcs_two = select_two_biggest_connected_region(rcs)
        cs_m_two = select_two_biggest_connected_region(cs_m)

        # 4. run discontinuity detection on rcs with cs_a and cs_dl -> arcs
        result_dict = self.discontinuity_detection_default(seg_array=rcs,
                                                           gt_array=None,
                                                           auxiliary_array=cs_a,
                                                           pre_mask_array=cs_dl,
                                                           spacing=self.spacing,
                                                           origin=self.origin,
                                                           direction=self.direction,
                                                           save_path=None,
                                                           )
        rcs_sphere = result_dict.get("mask_sphere")
        rcs_refined_num = result_dict.get("num")
        assert rcs_sphere is not None, "rcs_sphere is None"
        assert rcs_refined_num is not None, "rcs_num is None"
        arcs = self.merge(m1=rcs, m2=cs_a, dl=rcs_sphere)

        # 5. select_two_biggest_connected_region on arcs -> arcs_two
        arcs_two = select_two_biggest_connected_region(arcs)

        # 6. selectively merge rs and cs_m with cs_dl -> rcs_selected
        result_dict = self.discontinuity_detection_default(seg_array=cs_m,
                                                           gt_array=None,
                                                           auxiliary_array=rs,
                                                           pre_mask_array=cs_dl,
                                                           spacing=self.spacing,
                                                           origin=self.origin,
                                                           direction=self.direction,
                                                           save_path=None,
                                                           )
        cs_m_selected_sphere = result_dict.get("mask_sphere")
        cs_m_selected_refined_num = result_dict.get("num")
        assert cs_m_selected_sphere is not None, "rcs_selected_sphere is None"
        assert cs_m_selected_refined_num is not None, "rcs_selected_refined_num is None"
        rcs_selected = self.merge(m1=cs_m, m2=rs, dl=cs_m_selected_sphere)

        # 7. select_two_biggest_connected_region on rcs_select -> rcs_selected_two
        rcs_selected_two = select_two_biggest_connected_region(rcs_selected)

        # 8. merge

        arcs_selected = self.selected_final_merge(rcs_select=rcs_selected,
                                                  arcs=arcs,
                                                  dl1=cs_dl,
                                                  dl2=rcs_sphere)

        # 9. select_two_biggest_connected_region on arcs_selected -> arcs_selected_two
        arcs_selected_two = select_two_biggest_connected_region(arcs_selected)

        # 10. selectively merge rs and cs_m without cs_dl -> rcs_new
        result_dict = self.discontinuity_detection(seg_array=cs_m,
                                                   gt_array=None,
                                                   auxiliary_array=rs,
                                                   pre_mask_array=None,
                                                   spacing=self.spacing,
                                                   origin=self.origin,
                                                   direction=self.direction,
                                                   save_path=None,
                                                   )

        cs_m_new_sphere = result_dict.get("mask_sphere")
        cs_m_new_refined_num = result_dict.get("num")
        assert cs_m_new_sphere is not None, "rcs_new_sphere is None"
        assert cs_m_new_refined_num is not None, "rcs_new_refined_num is None"
        rcs_new = self.merge(m1=cs_m, m2=rs, dl=cs_m_new_sphere)

        # 11. select_two_biggest_connected_region on rcs_select -> rcs_selected_two
        rcs_new_two = select_two_biggest_connected_region(rcs_new)

        # 12. merge

        arcs_new = self.selected_final_merge(rcs_select=rcs_new,
                                             arcs=arcs,
                                             dl1=cs_dl,
                                             dl2=rcs_sphere)

        # 13. select_two_biggest_connected_region on arcs_selected -> arcs_selected_two
        arcs_new_two = select_two_biggest_connected_region(arcs_new)


        # 14. save
        self.save_image(rcs, self.save_subdir / f"{self.name}_RCS.nii.gz")
        self.save_image(rcs_two, self.save_subdir / f"{self.name}_RCS_TWO.nii.gz")
        self.save_image(arcs, self.save_subdir / f"{self.name}_ARCS.nii.gz")
        self.save_image(arcs_two, self.save_subdir / f"{self.name}_ARCS_TWO.nii.gz")
        self.save_image(rcs_selected, self.save_subdir / f"{self.name}_RCS_SELECTED.nii.gz")
        self.save_image(rcs_selected_two, self.save_subdir / f"{self.name}_RCS_SELECTED_TWO.nii.gz")
        self.save_image(cs_m_two, self.save_subdir / f"{self.name}_CS_M_TWO.nii.gz")
        self.save_image(arcs_selected, self.save_subdir / f"{self.name}_ARCS_SELECTED.nii.gz")
        self.save_image(arcs_selected_two, self.save_subdir / f"{self.name}_ARCS_SELECTED_TWO.nii.gz")
        self.save_image(rcs_new, self.save_subdir / f"{self.name}_RCS_NEW.nii.gz")
        self.save_image(rcs_new_two, self.save_subdir / f"{self.name}_RCS_NEW_TWO.nii.gz")
        self.save_image(arcs_new, self.save_subdir / f"{self.name}_ARCS_NEW.nii.gz")
        self.save_image(arcs_new_two, self.save_subdir / f"{self.name}_ARCS_NEW_TWO.nii.gz")

        # save little segment and sphere for visualization
        self.save_image(cs_m_selected_sphere, self.save_subdir / f"{self.name}_cs_m_selected_sphere.nii.gz")
        self.save_image(rcs_sphere, self.save_subdir / f"{self.name}_rcs_sphere.nii.gz")
        self.save_image(rcs * cs_dl, self.save_subdir / f"{self.name}_cs_m_merge_segment.nii.gz")
        self.save_image(cs_m_selected_sphere * rs, self.save_subdir / f"{self.name}_rcs_selected_merge_segment.nii.gz")
        self.save_image(rcs_sphere * cs_a, self.save_subdir / f"{self.name}_rcs_merge_segment.nii.gz")
        self.save_image(cs_m_new_sphere, self.save_subdir / f"{self.name}_cs_m_new_sphere.nii.gz")
        self.save_image(cs_m_new_sphere * rcs_new, self.save_subdir / f"{self.name}_rcs_new_merge_segment.nii.gz")

        # move cs_a, cs_m, cs_dl, rs, gt to save_subdir for visualization
        shutil.copy(self.cs_a_path, self.save_subdir / f"{self.name}_CS_A.nii.gz")
        shutil.copy(self.cs_m_path, self.save_subdir / f"{self.name}_CS_M.nii.gz")
        shutil.copy(self.cs_dl_path, self.save_subdir / f"{self.name}_CS_DL.nii.gz")
        shutil.copy(self.rs_path, self.save_subdir / f"{self.name}_RS.nii.gz")

        # save discontinuity metric
        d = {
            "name": self.name,
            "cs_m_selected_refined_num": cs_m_selected_refined_num,
            "arcs_improve": rcs_refined_num,
            "cs_m_new_refined_num": cs_m_new_refined_num,
        }
        return d

    def run_without_cs_a(self):
        """
        Run the post processing pipeline.
        1. load the data
        2. directly merge rs and cs_m with cs_dl -> rcs
        3. select_two_biggest_connected_region on rcs -> rcs_two
        4. run discontinuity detection on rcs with cs_a and cs_dl -> arcs
        5. select_two_biggest_connected_region on arcs -> arcs_two

        if selected merge:
        6. selectively merge rs and cs_m with cs_dl -> rcs_selected
        7. select_two_biggest_connected_region on rcs_select -> rcs_selected_two
        8. run final merge on rcs_select with cs_a and cs_dl -> arcs_selected
        9. select_two_biggest_connected_region on arcs_selected -> arcs_selected_two

        if new merge:
        10. selectively merge rs and cs_m without cs_dl -> rcs_new
        11. select_two_biggest_connected_region on rcs_new -> rcs_new_two
        12. run final merge on rcs_new with cs_a and cs_dl -> arcs_new
        13. select_two_biggest_connected_region on arcs_new -> arcs_new_two
        """
        cs_a, rs, cs_dl, cs_m = None, None, None, None
        # 1. load the data
        cs_m, spacing, origin, direction = self.read_image(self.cs_m_path)
        self.spacing, self.origin, self.direction = spacing, origin, direction

        rs, _, _, _ = self.read_image(self.rs_path)
        cs_dl, _, _, _ = self.read_image(self.cs_dl_path)

        rs_two = select_two_biggest_connected_region(rs)

        # 2. directly merge rs and cs_m with cs_dl -> rcs
        rcs = self.merge(m1=cs_m, m2=rs, dl=cs_dl)

        # 3. select_two_biggest_connected_region on rcs -> rcs_two
        rcs_two = select_two_biggest_connected_region(rcs)
        cs_m_two = select_two_biggest_connected_region(cs_m)

        # 6. selectively merge rs and cs_m with cs_dl -> rcs_selected
        result_dict = self.discontinuity_detection_default(seg_array=cs_m,
                                                           gt_array=None,
                                                           auxiliary_array=rs,
                                                           pre_mask_array=cs_dl,
                                                           spacing=self.spacing,
                                                           origin=self.origin,
                                                           direction=self.direction,
                                                           save_path=None,
                                                           )

        cs_m_selected_sphere = result_dict.get("mask_sphere")
        cs_m_selected_refined_num = result_dict.get("num")
        assert cs_m_selected_sphere is not None, "rcs_selected_sphere is None"
        assert cs_m_selected_refined_num is not None, "rcs_selected_refined_num is None"
        rcs_selected = self.merge(m1=cs_m, m2=rs, dl=cs_m_selected_sphere)

        # 7. select_two_biggest_connected_region on rcs_select -> rcs_selected_two
        rcs_selected_two = select_two_biggest_connected_region(rcs_selected)

        # 10. selectively merge rs and cs_m without cs_dl -> rcs_new
        result_dict = self.discontinuity_detection(seg_array=cs_m,
                                                   gt_array=None,
                                                   auxiliary_array=rs,
                                                   pre_mask_array=None,
                                                   spacing=self.spacing,
                                                   origin=self.origin,
                                                   direction=self.direction,
                                                   save_path=None,
                                                   )

        cs_m_new_sphere = result_dict.get("mask_sphere")
        cs_m_new_refined_num = result_dict.get("num")
        assert cs_m_new_sphere is not None, "rcs_new_sphere is None"
        assert cs_m_new_refined_num is not None, "rcs_new_refined_num is None"
        rcs_new = self.merge(m1=cs_m, m2=rs, dl=cs_m_new_sphere)

        # 11. select_two_biggest_connected_region on rcs_select -> rcs_selected_two
        rcs_new_two = select_two_biggest_connected_region(rcs_new)

        # 13. select_two_biggest_connected_region on arcs_selected -> arcs_selected_two


        # 14. save
        self.save_image(rcs, self.save_subdir / f"{self.name}_RCS.nii.gz")
        self.save_image(rcs_two, self.save_subdir / f"{self.name}_RCS_TWO.nii.gz")
        self.save_image(rcs_selected, self.save_subdir / f"{self.name}_RCS_SELECTED.nii.gz")
        self.save_image(rcs_selected_two, self.save_subdir / f"{self.name}_RCS_SELECTED_TWO.nii.gz")
        self.save_image(cs_m_two, self.save_subdir / f"{self.name}_CS_M_TWO.nii.gz")
        self.save_image(rcs_new, self.save_subdir / f"{self.name}_RCS_NEW.nii.gz")
        self.save_image(rcs_new_two, self.save_subdir / f"{self.name}_RCS_NEW_TWO.nii.gz")
        self.save_image(rs_two, self.save_subdir / f"{self.name}_RS_TWO.nii.gz")

        # save little segment and sphere for visualization
        self.save_image(cs_m_selected_sphere, self.save_subdir / f"{self.name}_cs_m_selected_sphere.nii.gz")
        self.save_image(cs_m_selected_sphere * rs, self.save_subdir / f"{self.name}_rcs_selected_merge_segment.nii.gz")
        self.save_image(cs_m_new_sphere, self.save_subdir / f"{self.name}_cs_m_new_sphere.nii.gz")

        # move cs_a, cs_m, cs_dl, rs, gt to save_subdir for visualization
        shutil.copy(self.cs_m_path, self.save_subdir / f"{self.name}_CS_M.nii.gz")
        shutil.copy(self.cs_dl_path, self.save_subdir / f"{self.name}_CS_DL.nii.gz")
        shutil.copy(self.rs_path, self.save_subdir / f"{self.name}_RS.nii.gz")

        # save discontinuity metric
        d = {
            "name": self.name,
            "cs_m_selected_refined_num": cs_m_selected_refined_num,
            "cs_m_new_refined_num": cs_m_new_refined_num,
        }
        return d


    def save_image(self, image, save_path, save_type=np.uint16):
        sitk_image = sitk.GetImageFromArray(image.astype(save_type))
        sitk_image.SetSpacing(self.spacing)
        sitk_image.SetOrigin(self.origin)
        sitk_image.SetDirection(self.direction)
        sitk.WriteImage(sitk_image, save_path)

    @staticmethod
    def merge(m1, m2, dl):
        m = m1 * (1 - dl) + m2 * dl
        return m

    @staticmethod
    def read_image(path):
        sitk_image = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(sitk_image)
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        return image, spacing, origin, direction

    @staticmethod
    def selected_final_merge(rcs_select, arcs, dl1, dl2):
        """TODO: selected_final_merge."""
        """
        :param rcs_select: path
        :param arcs: merge cs rs and a
        :param dl1: first stage discontinuity label
        :param dl2: second stage discontinuity label
        :param save_path: save path
        :param save_postfix: save postfix
        """
        if np.sum(dl2) != 0:
            id_map = cc3d.connected_components(dl1, connectivity=26)
            remain_id = np.unique(id_map[dl2 > 0])
            remain_index = np.zeros_like(dl1)
            for id in remain_id:
                if id > 0:
                    remain_index[id_map == id] = 1
            arcs_selected = rcs_select * (1 - remain_index) + arcs * remain_index
        else:
            arcs_selected = rcs_select

        return arcs_selected


class SparrowLinkDiscontinuityDetection:
    def __init__(self,
                 max_broken_size_percentage=0.1,
                 min_sphere_radius_percentage=0.015,
                 sphere_dilation_percentage=0.2,  # 0.1
                 region_select=4,
                 cube_min_size=(10, 10, 10),
                 skeleton_refine_times=3,
                 angle_threshold_1=0.0,
                 angle_threshold_2=0.0,
                 view=False,
                 region_threshold: Union[int, None] = 5,
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
                 seg_array: np.array = None,
                 gt_array: np.array = None,
                 auxiliary_array: np.array = None,
                 pre_mask_array: np.array = None,
                 spacing: Sequence[float] = None,
                 origin: Sequence[float] = None,
                 direction: Sequence[float] = None,
                 save_path: str = None
                 ):
        result_dict = discontinuity_detection(
            label=seg_array,
            GT=gt_array,
            auxiliary=auxiliary_array,
            pre_mask=pre_mask_array,
            spacing=spacing,
            direction=direction,
            origin=origin,
            save_path=save_path,
            view=self.view,
            max_broken_size_percentage=self.max_broken_size_percentage,
            min_sphere_radius_percentage=self.min_sphere_radius_percentage,
            sphere_dilation_percentage=self.sphere_dilation_percentage,
            region_select=self.region_select,
            cube_min_size=self.cube_min_size,
            skeleton_refine_times=self.skeleton_refine_times,
            region_threshold=self.region_threshold,
            angle_threshold_1=self.angle_threshold_1,
            angle_threshold_2=self.angle_threshold_2,
        )
        return result_dict


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
    parser.add_argument("--CS_M", type=str, default=None)
    parser.add_argument("--CS_DL", type=str, default=None)
    parser.add_argument("--RS", type=str, default=None)
    parser.add_argument("--CS_A", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_broken_size_percentage", type=float, default=0.3)
    parser.add_argument("--multiprocess", action='store_true', default=False)
    args = parser.parse_args()
    CS_M_path = pathlib.Path(args.CS_M)
    CS_DL_path = pathlib.Path(args.CS_DL)
    RS_path = pathlib.Path(args.RS)
    CS_A_path = pathlib.Path(args.CS_A) if args.CS_A is not None else None
    data_list = list(CS_M_path.glob("*.nii.gz"))
    print(f"\033[96m SparrowLink Postprocessing \033[00m")
    pbar = tqdm.tqdm(total=len(list(data_list)), colour="#87cefa")
    pbar.set_description("SparrowLink Processing")
    process_record = []

    if args.CS_A is not None:
        if args.multiprocess:
            poor = Pool(14)
        for file in data_list:
            name = file.name
            processor = SparrowLinkPostProcess(
                cs_m_path=str(CS_M_path / name),
                cs_dl_path=str(CS_DL_path / name),
                rs_path=str(RS_path / name),
                cs_a_path=str(CS_A_path / name),
                save_path=args.save_dir,
            )
            if not args.multiprocess:
                process_record.append(processor.run())
                pbar.update()
            else:
                poor.apply_async(
                    func=processor.run,
                    callback=lambda x: update(pbar, process_record, x),
                    error_callback=error_back
                )
        if args.multiprocess:
            poor.close()
            poor.join()

        process_record.sort(key=lambda x: x.get("name"))
        with open(args.save_dir + "/process_record.json", "w") as f:
            json.dump(process_record, f, indent=4)

    else:
        if args.multiprocess:
            poor = Pool(14)
        for file in data_list:
            name = file.name
            processor = SparrowLinkPostProcess(
                cs_m_path=str(CS_M_path / name),
                cs_dl_path=str(CS_DL_path / name),
                rs_path=str(RS_path / name),
                cs_a_path=None,
                save_path=args.save_dir,
                max_broken_size_percentage=args.max_broken_size_percentage,
            )
            if not args.multiprocess:
                process_record.append(processor.run_without_cs_a())
                pbar.update()
            else:
                poor.apply_async(
                    func=processor.run_without_cs_a,
                    callback=lambda x: update(pbar, process_record, x),
                    error_callback=error_back
                )
        if args.multiprocess:
            poor.close()
            poor.join()

        process_record.sort(key=lambda x: x.get("name"))
        with open(args.save_dir + "/process_record.json", "w") as f:
            json.dump(process_record, f, indent=4)








