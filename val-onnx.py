'''
Author: Assistant
Date: 2025-07-19
Description: 使用ONNX模型对验证集进行预测并保存性能报告
'''
import os
import time
import glob
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

class YOLOv8Detector:
    """YOLOv8 ONNX 模型推理类"""
    
    def __init__(self, onnx_model_path, conf_threshold=0.25, iou_threshold=0.45, device='cuda'):
        """
        初始化 YOLOv8 检测器
        
        Args:
            onnx_model_path: ONNX 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            device: 设备类型 ('cuda' 或 'cpu')
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # 检查可用的提供程序
        available_providers = ort.get_available_providers()
        print(f"系统可用的提供程序: {available_providers}")
        
        # 设置提供程序
        providers = []
        if device == 'cuda':
            if 'CUDAExecutionProvider' in available_providers:
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))
                print("✓ 使用 CUDA 加速")
            else:
                print("⚠ 警告: CUDAExecutionProvider 不可用，将使用 CPU")
                print("  请确保安装了 onnxruntime-gpu: pip install onnxruntime-gpu")
                providers.append('CPUExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')
            print("使用 CPU 模式")
        
        # 创建 ONNX Runtime 会话
        try:
            self.session = ort.InferenceSession(onnx_model_path, providers=providers)
            print(f"实际使用的提供程序: {self.session.get_providers()}")
        except Exception as e:
            print(f"创建 ONNX Runtime 会话失败: {e}")
            raise
        
        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 获取输入尺寸和数据类型
        input_info = self.session.get_inputs()[0]
        input_shape = input_info.shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.input_type = input_info.type
        
        # 确定数据类型
        if 'float16' in self.input_type:
            self.dtype = np.float16
            print("模型使用 float16 数据类型")
        else:
            self.dtype = np.float32
            print("模型使用 float32 数据类型")
        
        print(f"模型输入尺寸: {self.input_width}x{self.input_height}")
        print(f"输出节点数: {len(self.output_names)}")
        
        # 检查 GPU 信息
        if 'CUDAExecutionProvider' in self.session.get_providers():
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                print(f"\nGPU 信息:")
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    print(f"  GPU {i}: {name}")
                    print(f"  显存: {mem_info.used/1024**3:.1f}GB / {mem_info.total/1024**3:.1f}GB")
                pynvml.nvmlShutdown()
            except:
                print("无法获取 GPU 详细信息")
    
    def preprocess(self, image):
        """
        预处理图像
        
        Args:
            image: 输入图像 (BGR)
            
        Returns:
            预处理后的图像张量
        """
        # 保存原始尺寸
        self.orig_height, self.orig_width = image.shape[:2]
        
        # 调整图像大小
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # BGR 转 RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [0, 1] 并转换为正确的数据类型
        normalized = rgb.astype(self.dtype) / 255.0
        
        # 转换为 NCHW 格式
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # 添加批次维度
        batched = np.expand_dims(transposed, axis=0)
        
        # 确保连续性
        batched = np.ascontiguousarray(batched)
        
        return batched
    
    def postprocess(self, outputs):
        """
        后处理模型输出
        
        Args:
            outputs: 模型原始输出
            
        Returns:
            检测框列表，每个框为 (x1, y1, x2, y2, confidence, class_id)
        """
        # 获取输出
        if len(outputs) == 1:
            predictions = outputs[0]
        else:
            # 如果有多个输出，通常第一个是检测结果
            predictions = outputs[0]
        
        # 转换为float32以避免溢出
        predictions = predictions.astype(np.float32)
        
        # 检查输出形状
        print(f"输出形状: {predictions.shape}")
        
        # YOLOv8 输出格式可能是:
        # [1, 84, 8400] - (batch, channels, predictions)
        # [1, 8400, 84] - (batch, predictions, channels)
        
        if predictions.shape[1] > predictions.shape[2]:
            # [1, 8400, 84] 格式
            predictions = predictions[0]  # 移除批次维度
        else:
            # [1, 84, 8400] 格式
            predictions = np.transpose(predictions, (0, 2, 1))[0]
        
        boxes = []
        num_classes = predictions.shape[1] - 4  # 总通道数减去4个边界框参数
        
        for i, pred in enumerate(predictions):
            # 提取边界框坐标
            x_center = float(pred[0])
            y_center = float(pred[1])
            width = float(pred[2])
            height = float(pred[3])
            
            # 检查坐标是否有效
            if not all(np.isfinite([x_center, y_center, width, height])):
                continue
            
            # 类别概率
            class_probs = pred[4:]
            
            # 获取最高类别概率和对应的类别ID
            max_prob = float(np.max(class_probs))
            class_id = int(np.argmax(class_probs))
            
            # 过滤低置信度预测
            if max_prob < self.conf_threshold:
                continue
            
            # 限制坐标范围，避免溢出
            x_center = np.clip(x_center, 0, self.input_width)
            y_center = np.clip(y_center, 0, self.input_height)
            width = np.clip(width, 0, self.input_width)
            height = np.clip(height, 0, self.input_height)
            
            # 转换为角点坐标
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # 缩放到原始图像尺寸
            x1 = x1 * self.orig_width / self.input_width
            y1 = y1 * self.orig_height / self.input_height
            x2 = x2 * self.orig_width / self.input_width
            y2 = y2 * self.orig_height / self.input_height
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, self.orig_width - 1))
            y1 = max(0, min(y1, self.orig_height - 1))
            x2 = max(0, min(x2, self.orig_width - 1))
            y2 = max(0, min(y2, self.orig_height - 1))
            
            # 确保x2 > x1 且 y2 > y1
            if x2 <= x1 or y2 <= y1:
                continue
                
            boxes.append([int(x1), int(y1), int(x2), int(y2), max_prob, class_id])
        
        print(f"过滤前检测框数: {len(predictions)}, 过滤后: {len(boxes)}")
        
        # 应用非极大值抑制
        if boxes:
            boxes = self.nms(boxes)
            print(f"NMS后检测框数: {len(boxes)}")
        
        return boxes
    
    def nms(self, boxes):
        """
        非极大值抑制
        
        Args:
            boxes: 检测框列表
            
        Returns:
            NMS 后的检测框列表
        """
        if not boxes:
            return []
        
        # 转换为 numpy 数组
        boxes_array = np.array(boxes, dtype=np.float32)
        
        # 提取坐标和分数
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        scores = boxes_array[:, 4]
        
        # 计算面积
        areas = (x2 - x1) * (y2 - y1)
        
        # 按分数排序
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            # 保留IoU小于阈值的框
            inds = np.where(ovr <= self.iou_threshold)[0]
            order = order[inds + 1]
        
        return [boxes[i] for i in keep]
    
    def detect(self, image):
        """
        执行目标检测
        
        Args:
            image: 输入图像 (BGR)
            
        Returns:
            检测结果
        """
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 后处理
        boxes = self.postprocess(outputs)
        
        return boxes
    
    def draw_boxes(self, image, boxes):
        """
        在图像上绘制检测框
        
        Args:
            image: 原始图像
            boxes: 检测框列表
            
        Returns:
            绘制了检测框的图像
        """
        result_image = image.copy()
        
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            
            # 绘制边界框
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"Class {int(class_id)}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1_label = max(int(y1), label_size[1])
            cv2.rectangle(result_image, (int(x1), y1_label - label_size[1] - 4),
                         (int(x1) + label_size[0], y1_label), (0, 255, 0), -1)
            cv2.putText(result_image, label, (int(x1), y1_label - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result_image


def check_cuda_availability():
    """检查 CUDA 是否可用"""
    import subprocess
    
    # 检查 NVIDIA 驱动
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n✓ NVIDIA GPU 驱动已安装")
            # 提取部分信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA-SMI' in line or 'Driver Version' in line:
                    print(f"  {line.strip()}")
        else:
            print("\n✗ 未检测到 NVIDIA GPU 驱动")
            return False
    except:
        print("\n✗ nvidia-smi 命令不可用，可能未安装 NVIDIA 驱动")
        return False
    
    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch CUDA 可用: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("✗ PyTorch CUDA 不可用")
    except ImportError:
        print("  PyTorch 未安装，跳过 CUDA 检查")
    
    # 检查 onnxruntime-gpu
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        print("✓ ONNX Runtime GPU 支持已启用")
        return True
    else:
        print("✗ ONNX Runtime GPU 支持未启用")
        print("  请安装: pip install onnxruntime-gpu")
        return False


def main():
    """主函数"""
    # 配置参数
    onnx_model_path = "/home/simon/yolo-V11/runs/train/weights/best.onnx"
    val_image_folder = "/home/simon/yolo-V8/dataset/20xSpermYOLO_dataset/valid/images/"
    output_dir = "/home/simon/yolo-V11/runs/onnx_validation_results"
    use_cuda = True  # 是否使用 CUDA
    
    print("="*60)
    print("ONNX 模型验证程序")
    print("="*60)
    
    # 检查 CUDA 可用性
    if use_cuda:
        cuda_available = check_cuda_availability()
        if not cuda_available:
            print("\n警告: CUDA 不可用，将退出程序")
            print("如果要使用 CPU 模式，请设置 use_cuda = False")
            return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f"val_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 创建图像输出目录
    images_output_dir = os.path.join(run_dir, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    
    # 创建标签输出目录
    labels_output_dir = os.path.join(run_dir, "labels")
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # 初始化检测器
    print("\n正在加载 ONNX 模型...")
    try:
        device = 'cuda' if use_cuda else 'cpu'
        yolov8_detector = YOLOv8Detector(onnx_model_path, device=device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(val_image_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(val_image_folder, ext.upper())))
    
    if not image_paths:
        print(f"在 {val_image_folder} 中未找到图像文件")
        return
    
    print(f"\n找到 {len(image_paths)} 张图像")
    print("="*60)
    
    # 处理每张图像
    processing_times = []
    total_images = len(image_paths)
    failed_images = []
    
    # 预热 GPU
    if use_cuda and 'CUDAExecutionProvider' in yolov8_detector.session.get_providers():
        print("\n预热 GPU...")
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            _ = yolov8_detector.detect(dummy_image)
        print("GPU 预热完成")
        print("="*60)
    
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        print(f"\n处理图像 [{idx+1}/{total_images}]: {img_name}")
        
        try:
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                print(f"无法读取图像: {img_path}")
                failed_images.append(img_name)
                continue
            
            print(f"图像尺寸: {image.shape}")
            
            # 测量处理时间
            start_time = time.time()
            boxes = yolov8_detector.detect(image)
            end_time = time.time()
            
            # 计算处理时间（毫秒）
            processing_time_ms = (end_time - start_time) * 1000
            processing_times.append(processing_time_ms)
            
            print(f"检测到 {len(boxes)} 个目标, 用时: {processing_time_ms:.2f} ms")
            
            # 绘制检测结果
            result_image = yolov8_detector.draw_boxes(image, boxes)
            
            # 保存带检测框的图像
            output_image_path = os.path.join(images_output_dir, img_name)
            cv2.imwrite(output_image_path, result_image)
            
            # 保存检测结果为YOLO格式的txt文件
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            txt_path = os.path.join(labels_output_dir, txt_name)
            
            with open(txt_path, 'w') as f:
                h, w = image.shape[:2]
                for box in boxes:
                    x1, y1, x2, y2, conf, class_id = box
                    # 转换为YOLO格式 (class_id, x_center, y_center, width, height)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
        except Exception as e:
            print(f"处理图像 {img_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            failed_images.append(img_name)
            continue
    
    # 计算性能指标
    if processing_times:
        avg_time_ms = np.mean(processing_times)
        std_time_ms = np.std(processing_times)
        avg_fps = 1000 / avg_time_ms
        max_time_ms = max(processing_times)
        min_time_ms = min(processing_times)
        
        # 生成性能报告
        report_path = os.path.join(run_dir, "performance_report.txt")
        with open(report_path, 'w') as f:
            f.write("ONNX模型性能报告\n")
            f.write("="*50 + "\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型路径: {onnx_model_path}\n")
            f.write(f"推理提供程序: {yolov8_detector.session.get_providers()}\n")
            f.write(f"输入尺寸: {yolov8_detector.input_width}x{yolov8_detector.input_height}\n")
            f.write(f"输入数据类型: {yolov8_detector.input_type}\n")
            f.write(f"置信度阈值: {yolov8_detector.conf_threshold}\n")
            f.write(f"NMS IoU阈值: {yolov8_detector.iou_threshold}\n")
            f.write("-"*50 + "\n")
            f.write(f"总图片数: {total_images}\n")
            f.write(f"成功处理图片数: {len(processing_times)}\n")
            f.write(f"失败图片数: {len(failed_images)}\n")
            if failed_images:
                f.write(f"失败图片列表: {', '.join(failed_images)}\n")
            f.write(f"平均处理时间: {avg_time_ms:.2f} ± {std_time_ms:.2f} ms/图片\n")
            f.write(f"最长处理时间: {max_time_ms:.2f} ms\n")
            f.write(f"最短处理时间: {min_time_ms:.2f} ms\n")
            f.write(f"平均帧率: {avg_fps:.2f} FPS\n")
            f.write("="*50 + "\n")
        
        print("\n" + "="*60)
        print("验证完成！")
        print(f"结果保存在: {run_dir}")
        print(f"性能报告: {report_path}")
        print(f"成功处理: {len(processing_times)} 张图像")
        if failed_images:
            print(f"失败: {len(failed_images)} 张图像")
        print("="*60)
    else:
        print("没有成功处理的图像")


if __name__ == "__main__":
    main()