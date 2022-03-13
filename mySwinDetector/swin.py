import mmcv
from mmcv.runner import load_checkpoint
from pathlib import Path
import os

if not "root" in locals():
        current_path = Path(os.getcwd())
        root = current_path.parent.absolute()
os.chdir(root)

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

def load_swin():
    """
    Load the model from the checkpoint file and set the model to evaluation mode
    :return: The model is being returned.
    """


    print(root)
    config = 'jovyan/my_work/mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = 'jovyan/my_work/mmdetection/checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'

    # Set the device to be used for evaluation
    device='cuda:0'

    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']
    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    return model

if __name__ == '__main__':
    load_swin()