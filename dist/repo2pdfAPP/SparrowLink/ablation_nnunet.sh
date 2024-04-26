model="nnunetv2"
python3 modify_key_in_config.py \
--config_path=configs/two_stage2/second_stage/nnunetv2.yaml
#for i in 0 1
#do
#    sh SparrowLinkv3_nnUnet.sh -c 00111 -t 1 -s ${model}_new_${i}
#done
#sh SparrowLinkv3_nnUnet.sh -c 00111 -t 1 -s ${model}_new_I_M

#sh SparrowLinkv3_nnUnet.sh -c 00110 -t 0 -s ${model}_3
#sh SparrowLinkv3_nnUnet.sh -c 00110 -t 1 -s ${model}_4
#sh SparrowLinkv3_nnUnet.sh -c 00110 -t 1 -s ${model}_I_M


#sh SparrowLinkv3_nnUnet.sh -c 00011 -t 0 -s ${model}_0
#sh SparrowLinkv3_nnUnet.sh -c 00011 -t 0 -s ${model}_1
#sh SparrowLinkv3_nnUnet.sh -c 00011 -t 0 -s ${model}_2
#sh SparrowLinkv3_nnUnet.sh -c 00011 -t 0 -s ${model}_3
sh SparrowLinkv3_nnUnet.sh -c 00011 -t 0 -s ${model}_new_MMAD
sh SparrowLinkv3_nnUnet.sh -c 00011 -t 0 -s ${model}_new_no_a