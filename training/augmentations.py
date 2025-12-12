import albumentations as A
import cv2

# Although YOLOv8 has built-in augmentations, this file serves as a reference 
# for what "Strong Augmentations" typically look like in a custom pipeline.
# To integrate this into YOLOv8's dataloader requires hacking the dataset loader
# or pre-processing data offline, but YOLO's internal 'mosaic' and 'mixup' are usually sufficient.

def get_train_transforms(height, width):
    return A.Compose([
        A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(),
        ], p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

if __name__ == "__main__":
    print("Augmentation pipeline defined.")
