import sys
sys.path.append("/Users/ishirgarg/Github/UAV_Playground/model_finetuning")

from RGBNeonDataset import RGBNEONTreeDataset
from NEONTrainDataset import NEONTrainDataset
from PIL import Image
dataset = NEONTrainDataset("/Users/ishirgarg/Github/UAV_Playground/model_finetuning/UAV_NEON_cropped_images.pickle")

for i in range(len(dataset)):
    img = dataset[i]["rgb"]
    annotation = dataset[i]["annotation"]

    Image.fromarray(img).save(f"/Users/ishirgarg/Github/UAV_Playground/model_finetuning/YOLOv8/config/train/images/img_{i}.tif")
    with open(f"/Users/ishirgarg/Github/UAV_Playground/model_finetuning/YOLOv8/config/train/labels/img_{i}.txt", "w") as label_file:
        for bbox in annotation:
            bbox = bbox / 400 # Normalize
            label_file.write(f"0 {(bbox[0] + bbox[2]) / 2} {(bbox[1] + bbox[3]) / 2} {bbox[2] - bbox[0]} {bbox[3] - bbox[1]}\n")