"""
Author: SimonHanYANG SimonCK666@163.com
Date: 2025-07-18 10:12:11
LastEditors: SimonHanYANG SimonCK666@163.com
LastEditTime: 2025-08-02 11:05:37
FilePath: /yolo-V11/train.py
Description: 训练YOLOv8模型并进行验证、测试和导出ONNX模型.
"""

import glob
import os
import time

import numpy as np
import yaml

from ultralytics import YOLO

# 配置参数
CONFIG = {
    "data_yaml": "dataset/20xSpermYOLO_dataset/20XSpermYOLODatset.yaml",
    "epochs": 100,
    "img_size": 640,
    "batch_size": 16,
    "workers": 8,
    "device": 0,  # GPU设备，使用CPU设置为'cpu'
    "project": "runs/",
    "name": "train",
    "export_formats": ["onnx"],  # 导出的模型格式
    "validation_output": "validation_results/",  # 验证结果保存路径
}


def get_validation_images(data_yaml_path):
    """从数据YAML文件中获取验证集图片路径."""
    with open(data_yaml_path) as f:
        data_dict = yaml.safe_load(f)

    # 获取验证集路径
    val_images_dir = os.path.join(os.path.dirname(data_yaml_path), data_dict.get("val", ""))
    if not os.path.exists(val_images_dir):
        val_images_dir = os.path.join(data_dict.get("path", ""), data_dict.get("val", ""))

    # 如果val路径指向的是一个文件夹而不是images文件夹，则需要添加images
    if not val_images_dir.endswith("images"):
        val_images_dir = os.path.join(val_images_dir, "images")

    print(f"验证集图片路径: {val_images_dir}")

    # 获取验证集中的所有图片
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(val_images_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(val_images_dir, ext.upper())))

    return image_paths


def validate_model(model_path, image_paths, output_dir):
    """验证模型并测量每张图片的预测时间."""
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    processing_times = []

    print(f"\n开始验证模型: {model_path}")
    print(f"共有 {len(image_paths)} 张验证图片")

    for img_path in image_paths:
        img_name = os.path.basename(img_path)

        # 测量处理时间
        start_time = time.time()
        model.predict(
            source=img_path,
            save=True,
            save_txt=True,
            project=output_dir,
            name=os.path.basename(model_path).split(".")[0],
        )
        end_time = time.time()

        # 计算处理时间（毫秒）
        processing_time_ms = (end_time - start_time) * 1000
        processing_times.append(processing_time_ms)

        print(f"图片 {img_name} 处理完成 - 用时: {processing_time_ms:.2f} ms")

    # 计算性能指标
    avg_time_ms = np.mean(processing_times)
    std_time_ms = np.std(processing_times)
    avg_fps = 1000 / avg_time_ms
    max_time_ms = max(processing_times)
    min_time_ms = min(processing_times)

    # 输出性能报告
    print("\n性能报告:")
    print(f"总图片数: {len(image_paths)}")
    print(f"平均处理时间: {avg_time_ms:.2f} ± {std_time_ms:.2f} ms/图片")
    print(f"最长处理时间: {max_time_ms:.2f} ms")
    print(f"最短处理时间: {min_time_ms:.2f} ms")
    print(f"平均帧率: {avg_fps:.2f} FPS")

    # 将性能报告保存到文件
    report_path = os.path.join(output_dir, f"{os.path.basename(model_path).split('.')[0]}_performance_report.txt")
    with open(report_path, "w") as f:
        f.write("性能报告:\n")
        f.write(f"模型: {model_path}\n")
        f.write(f"总图片数: {len(image_paths)}\n")
        f.write(f"平均处理时间: {avg_time_ms:.2f} ± {std_time_ms:.2f} ms/图片\n")
        f.write(f"最长处理时间: {max_time_ms:.2f} ms\n")
        f.write(f"最短处理时间: {min_time_ms:.2f} ms\n")
        f.write(f"平均帧率: {avg_fps:.2f} FPS\n")

    print(f"性能报告已保存到 {report_path}")
    return model


def export_model(model, export_formats, output_dir):
    """导出模型为指定格式."""
    os.makedirs(output_dir, exist_ok=True)

    for format_name in export_formats:
        print(f"\n开始导出 {format_name} 格式模型...")
        try:
            model.export(format=format_name, imgsz=CONFIG["img_size"])
            print(f"{format_name} 格式模型导出成功!")
        except Exception as e:
            print(f"导出 {format_name} 格式模型失败: {str(e)}")


def main():
    """主函数，包括训练、验证和导出模型."""
    # 创建输出目录
    os.makedirs(CONFIG["validation_output"], exist_ok=True)

    # 加载预训练模型
    model = YOLO("yolo11n.pt")  # 加载预训练模型

    # 训练模型
    print("开始训练模型...")
    model.train(
        data=CONFIG["data_yaml"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["img_size"],
        batch=CONFIG["batch_size"],
        workers=CONFIG["workers"],
        device=CONFIG["device"],
        project=CONFIG["project"],
        name=CONFIG["name"],
    )

    # 获取训练后的模型路径
    train_dir = os.path.join(CONFIG["project"], CONFIG["name"])
    best_model_path = os.path.join(train_dir, "weights", "best.pt")
    last_model_path = os.path.join(train_dir, "weights", "last.pt")

    # 获取验证集图片路径
    val_images = get_validation_images(CONFIG["data_yaml"])

    if not val_images:
        print("未找到验证图片，请检查数据YAML文件和路径")
        return

    # 验证best模型
    best_model = validate_model(best_model_path, val_images, os.path.join(CONFIG["validation_output"], "best"))

    # 验证last模型
    last_model = validate_model(last_model_path, val_images, os.path.join(CONFIG["validation_output"], "last"))

    # 导出best模型为指定格式
    export_dir = os.path.join(train_dir, "exports")
    export_model(best_model, CONFIG["export_formats"], os.path.join(export_dir, "best"))

    # 导出last模型为指定格式
    export_model(last_model, CONFIG["export_formats"], os.path.join(export_dir, "last"))

    print("\n训练、验证和模型导出流程全部完成!")


if __name__ == "__main__":
    main()
