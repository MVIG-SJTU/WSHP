#!/bin/bash
pascal_pose_file_root="/path/to/pascal_pose_file.csv"
pascal_mask_img_dir="/path/to/pascal_mask_img"
origin_img_root="/path/to/origin_img"
json_file_root="/path/to/pose_json_file"
crop_output_path="/path/to/output/cropped_img_and_prior"
experiment_name="exp1"
merge_output_path="/path/to/output/merged_parsing_label"
overlay_output_path="/path/to/output/overlayed_image"

python crop_pose_and_generate_testing_prior.py --PASCALPoseFileRoot $pascal_pose_file_root --PASCALMaskImgDir $pascal_mask_img_dir --n 3 --k 3 --aug 0.25 --origin_img_root $origin_img_root --json_file_root $json_file_root --outputDir $crop_output_path
cd refinement_network
python3 test.py --dataroot $crop_output_path --dataset_mode single --model test --output_nc 1 --name $experiment_name --which_epoch latest
cd ..
python merge_parsing_result.py --outputDir $merge_output_path --parsing_root ./refinement_network/results/${experiment_name}/test_latest/images --origin_img_root $origin_img_root --json_file_root $json_file_root --aug 0.25
python overlay.py --origin_img_root $origin_img_root --parsing_img_root $merge_output_path --outputDir $overlay_output_path
