import cv2
import os
import csv
from ultralytics import YOLO

# --- Configuration ---
MODEL_WEIGHTS_PATH = "ppe_detection/exp_optimized17/weights/best.pt"
VIDEO_FILE_PATH = "./Videos/8964770-uhd_3840_2160_25fps.mp4"
CONFIDENCE_THRESHOLD = 0.65
IMG_SIZE = 1280
OUTPUT_CSV = "ppe_violations_log.csv"
SAVE_OUTPUT_VIDEO = True
SHOW_VIDEO = True

# Classes that indicate missing PPE
CLASS_MAP = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "Machinery",
    9: "Vehicle"
}

NON_COMPLIANT_CLASSES = [2, 3]  # NO-Hardhat, NO-Mask, NO-Safety Vest

def draw_boxes(frame, boxes, classes, confidences):
    # Updated colors (no white), distinct for each class (BGR)
    CLASS_COLORS = {
        0: (0, 255, 255),   # Hardhat - Yellow
        1: (255, 0, 255),   # Mask - Pink
        2: (0, 0, 255),     # NO-Hardhat - Red
        3: (0, 0, 200),     # NO-Mask - Dark Red
        4: (0, 64, 255),    # NO-Safety Vest - Orange-Red
        5: (0, 255, 0),     # Person - Green
        6: (255, 0, 0),     # Safety Cone - Blue
        7: (0, 165, 255),   # Safety Vest - Orange
        8: (128, 0, 128),   # Machinery - Purple
        9: (0, 128, 255)    # Vehicle - Sky Blue
    }

    for box, cls_id, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        label = f"{CLASS_MAP.get(cls_id, str(cls_id))}: {conf:.2f}"
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))  # Fallback = gray
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def perform_video_inference():
    # Load model
    print(f"Loading model from: {MODEL_WEIGHTS_PATH}")
    model = YOLO(MODEL_WEIGHTS_PATH)

    # Open video
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        print(f"❌ Could not open video file: {VIDEO_FILE_PATH}")
        return

    # Prepare video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if SAVE_OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("model_output_video.mp4", fourcc, fps, (width, height))

    # Prepare CSV log
    csv_file = open(OUTPUT_CSV, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        # Log violations
        for box, cls_id, conf in zip(boxes, classes, confidences):
            if cls_id in NON_COMPLIANT_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                csv_writer.writerow([frame_id, CLASS_MAP.get(cls_id, cls_id), round(conf, 2), x1, y1, x2, y2])

        # Draw boxes
        annotated_frame = draw_boxes(frame, boxes, classes, confidences)

        if SHOW_VIDEO:
            cv2.imshow("PPE Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if SAVE_OUTPUT_VIDEO:
            out.write(annotated_frame)

        frame_id += 1

    # Clean up
    cap.release()
    if SAVE_OUTPUT_VIDEO:
        out.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print(f"✅ Inference complete. Saved to 'annotated_output.mp4' and log to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    perform_video_inference()
