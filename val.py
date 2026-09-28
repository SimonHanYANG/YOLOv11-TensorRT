"""
Author: SimonHanYANG SimonCK666@163.com
Date: 2025-07-19 08:58:46
LastEditors: SimonHanYANG SimonCK666@163.com
LastEditTime: 2025-08-02 11:09:42
FilePath: /yolo-V8/val.py
Description: val code.
"""

import cv2

from ultralytics import YOLO

model = YOLO("/home/simon/yolo-V11/runs/detect/train/weights/best.pt")

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
# im1 = Image.open("/home/cj/chaintwork/yolov8/001.jpeg")
# results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
im2 = cv2.imread("/home/simon/yolo-V8/dataset/20xSpermYOLO_dataset/valid/images/pic20B-2.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
