from PIL import Image
import torch
import os

from NEONTrainDataset import NEONTrainDataset

pickle_path = '/Users/ishirgarg/Github/UAV_Playground/model_finetuning/NEON_train_crop400_v1.pickle'
train_path = '/Users/ishirgarg/Github/UAV_Playground/model_finetuning/YOLOv8/config/train'
val_path = '/Users/ishirgarg/Github/UAV_Playground/model_finetuning/YOLOv8/config/val'

full_dataset = NEONTrainDataset(pickle_path)

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [0.9, 0.1], torch.Generator().manual_seed(42))

BOX_THRESHOLD = 6

count = 0
def format_for_yolov8(dataset, data_path):
    global count
    '''Formats data from dataset into format compatible for YOLOv8'''
    for i in range(len(dataset)):
        img = dataset[i]["rgb"]
        annotation = dataset[i]["annotation"]

        Image.fromarray(img).save(os.path.join(data_path, f"images/image_{i}.tif"))
        with open(os.path.join(data_path, f"labels/image_{i}.txt"), "w") as label_file:                
            for bbox in annotation:
                if (bbox[2] - bbox[0] < BOX_THRESHOLD) or (bbox[3] - bbox[1] < BOX_THRESHOLD):
                    count += 1
                    continue
                bbox = bbox / img.shape[0] # Normalize (assumes image is square!)
                label_file.write(f"0 {(bbox[0] + bbox[2]) / 2} {(bbox[1] + bbox[3]) / 2} {bbox[2] - bbox[0]} {bbox[3] - bbox[1]}\n")

format_for_yolov8(train_dataset, train_path)
format_for_yolov8(val_dataset, val_path)

print(count)