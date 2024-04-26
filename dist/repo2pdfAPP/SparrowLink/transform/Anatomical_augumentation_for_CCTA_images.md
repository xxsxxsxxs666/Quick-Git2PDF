# Anatomy-Informed Data Augmentation for Coronary Artery in CCTA Image
* Video of the on-the-fly anatomy-based data augmentation （slice view）
![1709195253734 00h00m00s-00h00m07s](https://github.com/xxsxxsxxs666/SparrowLink/assets/61532031/5443c459-72b7-480e-8b30-06ff231b956d)

* Video of the on-the-fly anatomy-based data augmentation. （3D）
![3D Slicer 5 2 1 2024-02-29 15-57-46 00h00m00s-00h00m11s](https://github.com/xxsxxsxxs666/SparrowLink/assets/61532031/c203e8bf-a892-4227-98cb-060fc3c40671)

* real CCTA image video:
![1706926881079 00h00m00s-00h00m08s](https://github.com/xxsxxsxxs666/SparrowLink/assets/61532031/c6cdce80-186f-44d0-bcc9-4abb36e9f1ef)

* You can use our anatomy-based data augmentation tool by simply plugging it into MONAI transform architecture:

```python
save_transform = Compose(
        [
            LoadImaged(keys=["image", "label", "heart"]),
            EnsureChannelFirstd(keys=["image", "label", "heart"]),
            ArteryTransformD(keys=["image", "label"], image_key="image", artery_key="label", p_anatomy_per_sample=1,
                             p_contrast_per_sample=1,
                             contrast_reduction_factor_range=(0.6, 1), mask_blur_range=(3, 6),
                             mvf_scale_factor_range=(1, 2), mode=("bilinear", "nearest")),
            # HeartTransformD(keys=["image", "label", "heart"], artery_key="label", heart_key="heart",
            #                 p_anatomy_heart=0, p_anatomy_artery=1,
            #                 dil_ranges=((-10, 10), (-5, -3)), directions_of_trans=((1, 1, 1), (1, 1, 1)), blur=(32, 8),
            #                 mode=("bilinear", "nearest", "nearest"), visualize=True, batch_interpolate=True,
            #                 threshold=(-1, 0.5, 0.5)),
            # CASTransformD(keys=["image", "label", "heart"], label_key="label", heart_key="heart", p_anatomy_per_sample=1,
            #               dil_ranges=((-30, -40), (-300, -500)), directions_of_trans=((1, 1, 1), (1, 1, 1)), blur=[4, 32],
            #               mode=("bilinear", "nearest", "nearest"),),
            SaveImaged(keys=["image"], output_dir=save_dir, output_postfix='spatial_transform_image',
                       print_log=True, padding_mode="zeros"),
            SaveImaged(keys=["label"], output_dir=save_dir, output_postfix='spatial_transform_label',
                       print_log=True, padding_mode="zeros"),
        ]
    )
```

```python
save_transform = Compose(
        [
            LoadImaged(keys=["image", "label", "heart"]),
            EnsureChannelFirstd(keys=["image", "label", "heart"]),
            ArteryTransformD(keys=["image", "label"], image_key="image", artery_key="label", p_anatomy_per_sample=1,
                             p_contrast_per_sample=1,
                             contrast_reduction_factor_range=(0.6, 1), mask_blur_range=(3, 6),
                             mvf_scale_factor_range=(1, 2), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(128, 128, 128),
                pos=3,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            SaveImaged(keys=["image"], output_dir=save_dir, output_postfix='spatial_transform_image',
                       print_log=True, padding_mode="zeros"),
            SaveImaged(keys=["label"], output_dir=save_dir, output_postfix='spatial_transform_label',
                       print_log=True, padding_mode="zeros"),
        ]
    )
```
