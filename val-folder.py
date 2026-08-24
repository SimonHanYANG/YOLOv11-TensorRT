"""
Author: SimonHanYANG SimonCK666@163.com
Date: 2025-07-19 08:58:46
LastEditors: SimonHanYANG SimonCK666@163.com
LastEditTime: 2025-08-02 11:07:48
FilePath: /yolo-V8/val.py
Description: val code.
"""

import glob
import os
import sys
import time

from ultralytics import YOLO

# 加载模型
model = YOLO("/home/simon/yolo-V11/runs/train/weights/best.pt")

# 设置输入文件夹路径
input_folder = "/home/simon/yolo-V8/dataset/20xSpermYOLO_dataset/valid/images/"  # 替换为你的图片文件夹路径

# 设置输出文件夹路径
output_folder = "/home/simon/yolo-V11/runs/detect/predict/predict_all_imgs/"  # 替换为你想保存结果的文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有图片的路径
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))

total_images = len(image_paths)
if total_images == 0:
    print(f"错误：在 {input_folder} 中没有找到图片文件")
    sys.exit()

print(f"找到 {total_images} 张图片，开始处理...")

# 性能指标
processing_times = []

# 处理每张图片并记录时间
for img_path in image_paths:
    img_name = os.path.basename(img_path)

    # 测量处理时间
    start_time = time.time()
    result = model.predict(source=img_path, save=True, save_txt=True, project=output_folder, name="predictions")
    end_time = time.time()

    # 计算处理时间（毫秒）
    processing_time_ms = (end_time - start_time) * 1000
    processing_times.append(processing_time_ms)

    print(f"图片 {img_name} 处理完成 - 用时: {processing_time_ms:.2f} ms")

# 计算性能指标
avg_time_ms = sum(processing_times) / len(processing_times)
avg_fps = 1000 / avg_time_ms
max_time_ms = max(processing_times)
min_time_ms = min(processing_times)

# 输出性能报告
print("\n性能报告:")
print(f"总图片数: {total_images}")
print(f"平均处理时间: {avg_time_ms:.2f} ms/图片")
print(f"最长处理时间: {max_time_ms:.2f} ms")
print(f"最短处理时间: {min_time_ms:.2f} ms")
print(f"平均帧率: {avg_fps:.2f} FPS")
print(f"所有结果已保存到 {os.path.join(output_folder, 'predictions')}")

# 将性能报告保存到文件
report_path = os.path.join(output_folder, "performance_report.txt")
with open(report_path, "w") as f:
    f.write("性能报告:\n")
    f.write(f"总图片数: {total_images}\n")
    f.write(f"平均处理时间: {avg_time_ms:.2f} ms/图片\n")
    f.write(f"最长处理时间: {max_time_ms:.2f} ms\n")
    f.write(f"最短处理时间: {min_time_ms:.2f} ms\n")
    f.write(f"平均帧率: {avg_fps:.2f} FPS\n")

print(f"性能报告已保存到 {report_path}")
