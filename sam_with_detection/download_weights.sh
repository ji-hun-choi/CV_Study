#!/bin/zsh

wget https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth
mv detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv sam_vit_b_01ec64.pth weights/sam_vit_b_01ec64.pth


pip install git+https://github.com/facebookresearch/segment-anything.git