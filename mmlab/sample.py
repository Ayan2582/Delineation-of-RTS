import os
# Fix OMP: Error #15 (Duplicate OpenMP runtime crash on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
image_path = 'demo.jpg'

# 1. Initialize detector
model = init_detector(config_file, checkpoint_file, device='cpu')

# 2. Fix LocalVisBackend warning by assigning a output directory
model.cfg.visualizer.save_dir = 'vis_outputs'

# 3. Run inference
result = inference_detector(model, image_path)

# 4. Visualize and save the output image
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

image = mmcv.imread(image_path)
visualizer.add_datasample(
    'result',
    image,
    data_sample=result,
    draw_gt=False,
    wait_time=0,
    out_file='output.jpg'
)

print("Inference finished successfully! Output saved to output.jpg")