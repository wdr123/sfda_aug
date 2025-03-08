--------------------------------------------------------------------------------------------

Code for Text as a Bridge: Multi-Source Free Domain Adaptation for Object Detection

--------------------------------------------------------------------------------------------

This package contains source code for submission: Multi-Source Teacher-Student Knowledge-Fusion for Source-Free Domain
Adaptation in Object Detection

Dataset Preparation: Please download DWD dataset from: https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B

This code mainly contains two parts: text-based augmentation and Mean-Teacher CoTeaching.

PART I: text-based augmentation:
1.The text-based feature augmentation is based on PODA, please install the PODA package following https://github.com/astra-vision/PODA/tree/master
2.Add new augmentation file: poda/PIN_aug_R1.py to the PODA folder
3.Update main.py with poda/main.py
4.Add new dataset files: poda/dwd.py to the folder: PODA/datasets
5.Run example: CUDA_VISIBLE_DEVICES=1 python3 PIN_aug_R1.py --dataset night_sunny --data_root ./datasets/Night-Sunny --total_it 100 --resize_feat --save_dir ./night_sunny_aug/day_foggy --crop_size 320 --domain_desc "driving at foggy day times"

PART I can be conducted offline. 


PART II: Mean-Teacher CoTeaching
The code is based on MMDetection library. Please Install MMDetection following the instructions: https://mmdetection.readthedocs.io/en/latest/.

1.Update mmdetection/configs with aug_msfda/configs
2.Update mmdetection/mmdet with aug_msfda/mmdet
3.To run pre-trained source models: take day_clear as an example, run: bash ./tools/dist_train.sh configs/faster_rcnn/faster-rcnn_r50_fpn_1x_day_clear.py 4
4.Step 3 can be conduced offline. After obtaining augmentation data, put the ./night_sunny_aug to mmdetection/datasets and train the target model: take day_foggy as an example, run: bash ./tools/dist_train_one.sh configs/faster_rcnn/faster-rcnn_r50_fpn_1x_sourcefree_cot_Nrainy_Nclear_freeze_aug.py 2
