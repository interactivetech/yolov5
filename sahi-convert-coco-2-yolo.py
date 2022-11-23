from sahi_coco import Coco, export_coco_as_yolov5
from sahi.utils.file import load_json, save_json
COCO_DATASET_NAME = 'virat-aerial-156-frames-v2-coco'
coco = Coco.from_coco_dict_or_path(coco_dict_or_path=f'/Users/mendeza/Downloads/{COCO_DATASET_NAME}/annotations/instances_default.json',
                                   image_dir=f'/Users/mendeza/Downloads/{COCO_DATASET_NAME}/images/',
                                  remapping_dict={i:i-1 for i in [1,2]})
print(coco.categories)
export_coco_as_yolov5(
    output_dir=f'/Users/mendeza/Downloads/{COCO_DATASET_NAME}-yolov5/', 
    train_coco=coco, 
    val_coco=coco, 
    train_split_rate=0.2, 
    numpy_seed=1,
    mod_train_dir = f'/run/determined/workdir/{COCO_DATASET_NAME}-yolov5/',
    mod_val_dir = f'/run/determined/workdir/{COCO_DATASET_NAME}-yolov5/')