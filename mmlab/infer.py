import os
# Must be set before importing any PyTorch or OpenMMLab packages
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from mmdet.apis import DetInferencer 

inferencer = DetInferencer(model = 'yolox_tiny_8x8_300e_coco' , device = 'cpu')

out = inferencer('demo.jpg' , out_dir = 'outputs/' , pred_score_thr = 0.3)

pred = out['predictions'][0]
print(pred.keys())
print(pred['bboxes'][:3])