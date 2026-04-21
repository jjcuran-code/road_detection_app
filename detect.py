import sys
import json
from ultralytics import YOLO

# Classes as specified
CLASSES = [
    'pothole',
    'open_manhole',
    'good_road',
    'cracks',
    'S-severity',
    'A_severity',
    'B_severity',
    'C_severity'
]

def main():
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Usage: python detect.py <model_path> <image_path>'}))
        sys.exit(1)
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    model = YOLO(model_path)
    results = model(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if 0 <= cls_id < len(CLASSES):
                detections.append({
                    'class': CLASSES[cls_id],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                })
    print(json.dumps({'detections': detections}))

if __name__ == '__main__':
    main()
