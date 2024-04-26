# n represent network name and c represent the use which block, for example, c=1111 means use all three blocks
# t represent training controller, for example, t=1 means training process will be executed
while getopts "m:c:t:s:" opt;do
  case $opt in
  m)
    model=$OPTARG
    ;;
  c)
    CS=${OPTARG:0:1} # coarse segmentation
    IP=${OPTARG:1:1} # information processing
    RS=${OPTARG:2:1} # refined segmentation
    IF=${OPTARG:3:1} # final inference
    CM=${OPTARG:4:1} # calculate metric
    ;;
  t)
    train=${OPTARG}
    ;;
  s)
    second_stage_dir=${OPTARG}
  esac
done

echo "SparrowLink using backbone: [${model}]"
echo "coarse segmentation=[${CS}]"
echo "information processing=[${IP}]"
echo "refined segmentation=[${RS}]"
echo "final inference=[${IF}]"
echo "calculate metric=[${CM}]"
echo "train=[${train}]"
if [ ${second_stage_dir} == "1" ]; then
  second_stage_dir=${model}
fi
echo "second_stage_dir=[${second_stage_dir}]"


experiments_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/experiments/Graduate_project/two_stage2"
data_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/data/multi_phase_select/final_version_crop_train"
configs_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/configs/two_stage2"

############################################################# coarse segmentation ################################################################
### 1. train
if [ ${CS} == "1" ]; then
if [ ${train} == "1" ]; then
python3 main.py \
--mode=train \
--configs=${configs_path}/first_stage/${model}.yaml \
--experiments_path=${experiments_path}/first_stage/${model} \
--persist_path=${experiments_path}/first_stage/persist_cache \
--img_path=${data_path}/train/main/img_crop \
--label_path=${data_path}/train/main/label_crop_clip \
--val_set=${data_path}/train/main/val.txt \
--train_set=${data_path}/train/main/train.txt || { echo 'Error during training.'; exit 1; }
fi
#### 2. inference
for mode in "train" "test" ;
do
  for phase in "main" "auxiliary" ;
  do
    python3 main.py \
    --mode=infer \
    --configs=${configs_path}/first_stage/${model}.yaml \
    --experiments_path=${experiments_path}/first_stage/${model} \
    --img_path=${data_path}/${mode}/${phase}/img_crop \
    --output_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer \
    --pretrain_weight_path=${experiments_path}/first_stage/${model}/checkpoint/best_metric_model.pth || { echo 'Error during inference.'; exit 1; }
  done
done
#### 3. caculate metrics
## main test, GT is available in main phase
python3 caculate_metric.py \
--seg_path=${experiments_path}/first_stage/${model}/main_test_infer \
--label_path=${data_path}/test/main/label_crop_clip || { echo 'Error during caculating metrics.'; exit 1; }
fi
######################################################### information processing block ##################################################33##########
if [ ${IP} == "1" ]; then
### 1. fracture detection, fracture detection output is saved in the same folder as the input, named as
## args.root_path / f'{args.model}' / "fracture_detection" / f'{args.inference_stage}'
## like args.root_path/${model}/fracture_detection/args.inference_stage
for mode in "train" "test" ;
do
  for phase in "main" "auxiliary" ;
  do
    python3 post_processing/fracture_detection.py \
    --detection_stage=1 \
    --save_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_detection \
    --S_M=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer \
    --GT=${data_path}/${mode}/${phase}/label_crop_clip \
    --max_broken_size_percentage=0.12 \
    --multi_process \
    --view
  done
done

# move the discontinuity label for training
for mode in "train" "test" ;
do
  for phase in "main" "auxiliary" ;
  do
    for postfix in "_sphere" "_sphere_GT" ;
    do
      echo /first_stage/${model}/${phase}_${mode}_infer_detection
      python3 post_processing/move_file.py \
      --data_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_detection \
      --save_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer${postfix} \
      --hierarchical=/* \
      --data_postfix=${postfix} || { echo 'Error during discontinuity detection.'; exit 1; }
    done
  done
done

#### 2. registration, registration output is saved in the same folder as the input, named as
### f"{args.net}/auxiliary_{args.mode}_infer_{args.reg_algorithm}_reg_{args.time}"
for mode in "train" "test" ;
do
  python3 registration/label_registration.py \
  --reg_algorithm=SyNRA \
  --time=1 \
  --save_root=${experiments_path}/first_stage/${model}/registration \
  --main_mask_path=${experiments_path}/first_stage/${model}/main_${mode}_infer \
  --auxiliary_mask_path=${experiments_path}/first_stage/${model}/auxiliary_${mode}_infer \
  --main_img_path=${data_path}/${mode}/main/img_crop \
  --auxiliary_img_path=${data_path}/${mode}/auxiliary/img_crop \
  --mode=${mode}|| { echo 'Error during registration.'; exit 1; }
done
fi
############################################################### refined segmentation #############################################################################
if [ ${RS} == "1" ]; then
if [ ${train} == "1" ]; then
# 1. train
python3 second_stage_main.py \
--mode=train \
--iters=150 \
--configs=${configs_path}/second_stage/${second_stage_dir}.yaml \
--experiments_path=${experiments_path}/second_stage/${second_stage_dir} \
--persist_path=${experiments_path}/second_stage/${second_stage_dir}/persist_cache \
--dataset_information=${experiments_path}/first_stage/dataset_properties.json \
--label_path=${data_path}/train/main/label_crop_clip \
--val_set=${data_path}/train/main/val.txt \
--train_set=${data_path}/train/main/train.txt \
--select_file=${experiments_path}/first_stage/${model}/main_train_infer_detection/cube_gt.txt \
--I_M=${data_path}/train/main/img_crop \
--CS_M=${experiments_path}/first_stage/${model}/main_train_infer \
--CS_DL=${experiments_path}/first_stage/${model}/main_train_infer_sphere \
--CS_DLGT=${experiments_path}/first_stage/${model}/main_train_infer_sphere_GT \
--I_A=${experiments_path}/first_stage/${model}/registration/auxiliary_train_img_SyNRA_reg_1 \
--CS_A=${experiments_path}/first_stage/${model}/registration/auxiliary_train_infer_SyNRA_reg_1 \
--CS_W=${experiments_path}/first_stage/${model}/checkpoint/best_metric_model.pth || { echo 'Error during training of refined segmentation.'; exit 1; }
fi
## 2. inference
## use test mode
for phase in "main" "auxiliary" ; # "main" "auxiliary" ;
do
  auxiliary_phase=""
  mode="test"
  if [ ${phase} == "main" ]; then
    auxiliary_phase="auxiliary"
  else auxiliary_phase="main"
  fi
  python3 second_stage_main.py \
  --mode=infer \
  --configs=${configs_path}/second_stage/${second_stage_dir}.yaml \
  --experiments_path=${experiments_path}/second_stage/${second_stage_dir} \
  --output_path=${experiments_path}/second_stage/${second_stage_dir}/${phase}_${mode}_only_infer \
  --persist_path=${experiments_path}/second_stage/${second_stage_dir}/persist_cache \
  --dataset_information=${experiments_path}/first_stage/dataset_properties.json \
  --I_M=${data_path}/${mode}/${phase}/img_crop \
  --CS_M=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer \
  --CS_DL=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_sphere \
  --CS_DLGT=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_sphere_GT \
  --I_A=${experiments_path}/first_stage/${model}/registration/${auxiliary_phase}_${mode}_img_SyNRA_reg_1 \
  --CS_A=${experiments_path}/first_stage/${model}/registration/${auxiliary_phase}_${mode}_infer_SyNRA_reg_1 \
  --pretrain_weight_path=${experiments_path}/second_stage/${second_stage_dir}/checkpoint/best_metric_model.pth || { echo 'Error during inferring of refined segmentation.'; exit 1; }
done
fi
################################################################## inference block #########################################################################
### detection save direct merge result
### selected_detection save merge result which is selected the strategy that only successful RS will be selected to be merged
## 1 means discontinuity detection without GT within first stage
## 2 means discontinuity detection with GT within first stage
## postfix for searching: glob function will search for files with {postfix}.nii.gz, RCS and RCSGT might conflict, so we use _RCS
if [ ${IF} == "1" ]; then

for phase in "main" "auxiliary" ;
do
  auxiliary_phase=""
  mode="test"
  if [ ${phase} == "main" ]; then
    auxiliary_phase="auxiliary"
  else auxiliary_phase="main"
  fi
  python3 SparrowLink_Post_Process.py \
  --CS_M=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer \
  --CS_A=${experiments_path}/first_stage/${model}/registration/${auxiliary_phase}_${mode}_infer_SyNRA_reg_1 \
  --CS_DL=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_sphere \
  --RS=${experiments_path}/second_stage/${second_stage_dir}/${phase}_${mode}_only_infer \
  --save_dir=${experiments_path}/second_stage/${second_stage_dir}/${phase}_${mode}_infer_detection_1 \
  --max_broken_size_percentage=0.12 \
  --multiprocess || { echo 'Error during post processing'; exit 1; }
done

for phase in "main" "auxiliary" ;
do
  auxiliary_phase=""
  mode="test"
  if [ ${phase} == "main" ]; then
    auxiliary_phase="auxiliary"
  else auxiliary_phase="main"
  fi
  python3 SparrowLink_Post_Process.py \
  --CS_M=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer \
  --CS_A=${experiments_path}/first_stage/${model}/registration/${auxiliary_phase}_${mode}_infer_SyNRA_reg_1 \
  --CS_DL=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_sphere_GT \
  --RS=${experiments_path}/second_stage/${second_stage_dir}/${phase}_${mode}_only_infer \
  --save_dir=${experiments_path}/second_stage/${second_stage_dir}/${phase}_${mode}_infer_detection_2 \
  --max_broken_size_percentage=0.12 \
  --multiprocess || { echo 'Error during post processing'; exit 1; }
done

python3 post_processing/move_file.py \
--data_path=${data_path}/test/main/label_crop_clip \
--save_path=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_1 \
--save_postfix="_GT" \
--separate_folder || { echo 'Error during moving file.'; exit 1; }

python3 post_processing/move_file.py \
--data_path=${data_path}/test/main/label_crop_clip \
--save_path=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_2 \
--save_postfix="_GT" \
--separate_folder || { echo 'Error during moving file.'; exit 1; }

fi
########### Metric Calculation ##########
if [ ${CM} == "1" ]; then
time1=$(date)
echo ${time1}

export OUTDATED_IGNORE=1
for postfix in "CS_M" "_RCS" "_ARCS" "_ARCS_TWO" "_CS_M_TWO" "_RCS_SELECTED" "_RCS_SELECTED_TWO" "_ARCS_SELECTED" "_ARCS_SELECTED_TWO" "_RCS_NEW" "_RCS_NEW_TWO" "_ARCS_NEW" "_ARCS_NEW_TWO" ;
do
  python3 caculate_metric.py \
  --seg_path=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_1 \
  --seg_find=*/*${postfix}.nii.gz \
  --label_path=${data_path}/test/main/label_crop_clip \
  --metric_result_path=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_1/${postfix}_metric.xlsx \
  --multiprocess || { echo 'Error during calculating metric.'; exit 1; }
done

for postfix in "CS_M" "_RCS" "_ARCS" "_ARCS_TWO" "_CS_M_TWO" "_RCS_SELECTED" "_RCS_SELECTED_TWO" "_ARCS_SELECTED" "_ARCS_SELECTED_TWO" "_RCS_NEW" "_RCS_NEW_TWO" "_ARCS_NEW" "_ARCS_NEW_TWO" ;
do
  python3 caculate_metric.py \
  --seg_path=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_2 \
  --seg_find=*/*${postfix}.nii.gz \
  --label_path=${data_path}/test/main/label_crop_clip \
  --metric_result_path=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_2/${postfix}_metric.xlsx \
  --multiprocess || { echo 'Error during calculating metric.'; exit 1; }
done

for postfix in "CS_M" "_RCS" "_ARCS" "_RCS_SELECTED" "_ARCS_SELECTED" "_RCS_NEW" "_ARCS_NEW" ;
do
  python3 SparrowLink_metric.py \
  --seg=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_1 \
  --seg_find=*/*${postfix}.nii.gz \
  --gt=${data_path}/test/main/label_crop_clip \
  --metric_postfix=${postfix} \
  --multiprocess || { echo 'Error during calculating metric.'; exit 1; }
done

for postfix in "CS_M" "_RCS" "_ARCS" "_RCS_SELECTED" "_ARCS_SELECTED" "_RCS_NEW" "_ARCS_NEW" ;
do
  python3 SparrowLink_metric.py \
  --seg=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_2 \
  --seg_find=*/*${postfix}.nii.gz \
  --gt=${data_path}/test/main/label_crop_clip \
  --metric_postfix=${postfix} \
  --multiprocess || { echo 'Error during calculating metric.'; exit 1; }
done

for postfix in "_rcs_new_gt_sphere_segment" "_CS_M_CS_M_gt_sphere_segment";  # ablation study
do
  python3 caculate_metric.py \
  --seg_path=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_1 \
  --seg_find=*/*${postfix}.nii.gz \
  --label_path=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_1 \
  --label_find=*/*_sphere_gt_segment.nii.gz \
  --metric_result_path=${experiments_path}/second_stage/${second_stage_dir}/main_test_infer_detection_1/${postfix}_ablation_gt_color_metric.xlsx \
  --multiprocess || { echo 'Error during calculating metric.'; exit 1; }
done

time1=$(date)
echo ${time1}

fi
