# BIOM - Assignment 2: Face Detection Comparison

This project was created for the Biometrics course as Assignment 2. It compares several face detection approaches — Haar Cascade, a pretrained YOLOv8 face model, a custom fine-tuned YOLOv8 model, and the default (COCO-pretrained) YOLOv8 model — on a labeled face dataset, and includes a live webcam demo.

## Contents

- [`Zadanie2.py`](Zadanie2.py) — main script: dataset splitting, model training, detector evaluation (Precision / Recall / F1, IoU-based matching), and result visualization (image collage with ground-truth boxes, training curves, comparison bar charts).
- [`webcam.py`](webcam.py) — standalone live webcam demo using OpenCV's Haar Cascade face detector.
- [`yolo-faces.yaml`](yolo-faces.yaml) — Ultralytics dataset config used for training the custom YOLOv8 face model.
- [`yolov8n-face-lindevs.pt`](yolov8n-face-lindevs.pt) — pretrained YOLOv8n face-detection weights.

## Detectors compared

1. **Haar Cascade** (`cv2.CascadeClassifier`, OpenCV built-in frontal face model)
2. **YOLOv8 (pretrained face model)** — `yolov8n-face-lindevs.pt`
3. **Custom YOLOv8** — fine-tuned on the project's own dataset (`runs/train/faces/custom2/weights/best.pt`)
4. **Default YOLOv8** — `yolov8n.pt` (general COCO-pretrained model, not face-specific)

Each detector is evaluated with IoU-based matching against ground-truth bounding boxes to compute **Precision**, **Recall**, and **F1-score**.

## Requirements

- Python 3.9+
- Packages: `opencv-python`, `matplotlib`, `numpy`, `tqdm`, `ultralytics`, `torch-directml`, `onnxruntime`, `scikit-learn`, `pandas`, `Pillow`

Install with:

```bash
pip install opencv-python matplotlib numpy tqdm ultralytics torch-directml onnxruntime scikit-learn pandas Pillow
```

> `torch-directml` is used to run inference on DirectML-compatible GPUs on Windows. If you don't have a compatible GPU or are on a different OS, adjust the device setup in `Zadanie2.py` accordingly (e.g. use `"cpu"` or CUDA).

## Dataset layout

The scripts expect a YOLO-format dataset:

```
Dataset/
├── images/   # .jpg images
└── labels/   # .txt YOLO-format annotations (same filename as the image)
```

`split_dataset()` splits this into `data_split/{train,val,test}` (default: 70% / 20% / 10%).

## Usage

Run the main evaluation pipeline:

```bash
python Zadanie2.py
```

This will:

1. Show a collage of sample images with their ground-truth annotations.
2. Split the dataset into train/val/test sets.
3. Train the custom YOLOv8 face model (`train_custom_yolo`).
4. Run evaluation for all four detectors and print Precision/Recall/F1.
5. Plot training loss/metric curves and a Precision/Recall/F1 comparison chart.

Run the live webcam demo (Haar Cascade only):

```bash
python webcam.py
```

Press **Q** to quit the webcam window.

## Notes

- Set `USE_PREPROCESSING = True` in `Zadanie2.py` to enable CLAHE-based contrast enhancement before detection.
- Training is configured for 1 epoch on CPU by default (`train_custom_yolo`) — adjust `epochs`, `batch`, and `device` as needed.
