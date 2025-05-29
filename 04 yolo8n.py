from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")
    data_yaml = "final_dataset/dataset.yaml"

    model.train(
        data=data_yaml,
        epochs=500,
        imgsz=640,
        batch=16,               # adjust for your GPU
        device=0,
        workers=10,
        project="ppe_detection",
        name="exp_optimized",
        # optimizer tweaks
        optimizer="SGD",
        lr0=0.01,               # initial LR
        lrf=0.2,                # final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=5e-4,
        # LR schedule & warmup
        cos_lr=True,            # cosine decay
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # mixed precision
        amp=True,
        # multi-scale training
        multi_scale=False,
        cache="disk",
        # augmentations
        mosaic=0.25,
        mixup=0.1,
        flipud=0.1,
        fliplr=0.5,
        # save & val options are on by default
    )

if __name__ == "__main__":
    train_model()
