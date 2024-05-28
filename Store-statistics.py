import warnings
import time
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision.draw.color import Color
import yaml
from datetime import datetime

# 忽略弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 读取 store_name 和 store_id 的函数
def get_store_name(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['store_name']


def get_store_id(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['store_id']


# 配置文件路径
store_name_config_file = 'store_configs/store_name.yaml'
store_name = get_store_name(store_name_config_file)
store_id = get_store_id(store_name_config_file)

# 载入预训练模型
model = YOLO('checkpoint/yolov8n.pt')

# 越线检测位置
LINE_START = sv.Point(1, 320)  # 左起点
LINE_END = sv.Point(640, 420)  # 右终点
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

# 线的可视化配置
line_color = Color(r=224, g=57, b=151)
line_annotator = sv.LineZoneAnnotator(thickness=5, text_thickness=2, text_scale=1, color=line_color)

# 目标检测可视化配置
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

# 从USB摄像头获取实时图像
cap = cv2.VideoCapture(0)  # 0表示第一个摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "inference/output/output_from_camera.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# 打印视频尺寸大小
print(f"视频尺寸: {frame_width}x{frame_height}")

print("开始处理实时视频流...")

# 保存结果的目录和文件
output_dir = 'inference/output'  # 要保存到的文件夹
output_file = f'{output_dir}/number.yaml'

# 初始化计数
previous_in_count = 0
previous_out_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法从摄像头读取帧")
        break

    # 用YOLO模型进行目标跟踪
    results = model.track(source=frame, show=False, stream=True, verbose=False, device='cpu')

    for result in results:
        frame = result.orig_img

        # 用 supervision 解析预测结果
        detections = sv.Detections.from_ultralytics(result)

        # 过滤掉非行人类别（假设行人类别 ID 为 0）
        detections = detections[detections.class_id == 0]

        # 确保有检测结果才解析追踪ID
        if len(detections) > 0 and result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            # 获取每个目标的：追踪ID、类别名称、置信度
            class_ids = detections.class_id  # 类别ID
            confidences = detections.confidence  # 置信度
            tracker_ids = detections.tracker_id  # 多目标追踪ID
            labels = ['#{} {} {:.1f}'.format(tracker_ids[i], model.names[class_ids[i]], confidences[i] * 100) for i in
                      range(len(class_ids))]

            # 绘制目标检测可视化结果
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # 越线检测
            try:
                line_counter.trigger(detections=detections)
            except IndexError as e:
                print(f"IndexError: {e}")
                # print(f"detections: {detections}")
                print(f"all_anchors shape: {result.boxes.xywh.cpu().numpy().shape}")
            line_annotator.annotate(frame=frame, line_counter=line_counter)

            out.write(frame)

    # 记录跨线进入和离开变化的记录
    current_in_count = line_counter.in_count
    current_out_count = line_counter.out_count

    if current_in_count != previous_in_count or current_out_count != previous_out_count:
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        with open(output_file, 'a') as file:
            file.write(f"{store_id} {store_name} {current_time} {current_in_count} {current_out_count}\n")
        previous_in_count = current_in_count
        previous_out_count = current_out_count

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
out.release()
cap.release()

print('视频已保存', output_path)
print(f'结果已保存到 {output_file}')
