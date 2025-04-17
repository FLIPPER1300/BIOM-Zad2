import cv2
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch_directml
import onnxruntime as ort
import shutil
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

# ======= Konfigurácia =======
image_dir = "Dataset/images"  # obrázky vo formáte .jpg alebo .png
annotation_dir = "Dataset/labels"  # .txt súbory rovnakého mena ako obrázky
USE_PREPROCESSING = False  # Ak áno, použije sa CLAHE na zlepšenie kontrastu obrázkov

# Nastavenie zariadenia s DirectML (ak torch-directml je nainštalovaný)
device = torch_directml.device()


# ======= Inicializácia detektorov =======
haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
yolo_model = YOLO("yolov8n-face-lindevs.pt")
#yolo_model.export(format="onnx", opset=12, simplify=True, dynamic=True)
yolo_model.to(device)
#onnx_session = ort.InferenceSession("yolov8n-face-lindevs.onnx", providers=["DmlExecutionProvider"])
custom_yolo_model = YOLO("runs/train/faces/custom2/weights/best.pt")
custom_yolo_model.to(device)
default_yolo_model = YOLO("yolov8n.pt")
default_yolo_model.to(device)



# ======= Funkcie =======
def split_dataset(image_dir, label_dir, output_base='data_split', train_ratio=0.7, val_ratio=0.2):
    os.makedirs(output_base, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, 'labels'), exist_ok=True)

    all_images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    random.shuffle(all_images)

    train_end = int(len(all_images) * train_ratio)
    val_end = train_end + int(len(all_images) * val_ratio)

    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    for split, files in splits.items():
        for f in files:
            img_src = os.path.join(image_dir, f)
            lbl_src = os.path.join(label_dir, f.replace(".jpg", ".txt"))
            img_dst = os.path.join(output_base, split, 'images', f)
            lbl_dst = os.path.join(output_base, split, 'labels', f.replace(".jpg", ".txt"))
            shutil.copy(img_src, img_dst)
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, lbl_dst)

    return os.path.join(output_base, 'train'), os.path.join(output_base, 'val'), os.path.join(output_base, 'test')

def picture_collage(image_dir, annotation_dir):

    # Získaj všetky .jpg súbory a náhodne vyber 6
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    selected_images = random.sample(all_images, 6)

    plt.figure(figsize=(15, 10))

    for idx, img_name in enumerate(selected_images):
        # Cesty k obrázku a anotácii
        image_path = os.path.join(image_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        annotation_path = os.path.join(annotation_dir, txt_name)

        # Načítaj obrázok
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Načítaj anotácie, ak existujú
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, box_w, box_h = map(float, parts)

                        # Prepočítaj na pixely
                        x_center *= w
                        y_center *= h
                        box_w *= w
                        box_h *= h

                        # Vypočítaj roh bounding boxu
                        x1 = int(x_center - box_w / 2)
                        y1 = int(y_center - box_h / 2)
                        x2 = int(x_center + box_w / 2)
                        y2 = int(y_center + box_h / 2)

                        # Vykresli obdĺžnik
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Zobraz obrázok v koláži
        plt.subplot(2, 3, idx + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(img_name)

    plt.tight_layout()
    plt.show()

def preprocess_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_BGR2LAB)

def yolo_to_bbox(yolo_line, img_w, img_h):
    _, cx, cy, w, h = map(float, yolo_line.strip().split())
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [x1, y1, x2, y2]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def evaluate_detections(pred_boxes, gt_boxes, iou_thresh=0.7):
    TP = 0
    matched = set()
    for pb in pred_boxes:
        for i, gb in enumerate(gt_boxes):
            if i in matched: continue
            iou = compute_iou(pb, gb)
            if iou >= iou_thresh:
                TP += 1
                matched.add(i)
                break
    FP = len(pred_boxes) - TP
    FN = len(gt_boxes) - TP
    return TP, FP, FN

def evaluate_model(model, test_img_dir, test_label_dir, iou_thresh=0.7):
    TP = FP = FN = 0

    image_files = [f for f in os.listdir(test_img_dir) if f.endswith(".jpg")]

    for f in tqdm(image_files):
        img_path = os.path.join(test_img_dir, f)
        label_path = os.path.join(test_label_dir, f.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        h, w = img.shape[:2]

        # GT boxy
        with open(label_path, 'r') as f_txt:
            gt_boxes = [yolo_to_bbox(line, w, h) for line in f_txt.readlines()]

        # Predikcia
        results = model(img)
        pred_boxes = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            pred_boxes.append([int(x) for x in box[:4]])

        # Vyhodnotenie
        tp, fp, fn = evaluate_detections(pred_boxes, gt_boxes, iou_thresh)
        TP += tp
        FP += fp
        FN += fn

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    print("\n📊 Vyhodnotenie vlastného modelu:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

def plot_conf_matrix(tp, fp, fn):
    cm = np.array([[tp, fp], [fn, 0]])  # TN nevieme pri detekcii
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Detected", "Missed"])
    disp.plot()
    plt.title("Konfúzna matica (TP, FP, FN)")
    plt.show()

def compare_results(results_dict):
    df = pd.DataFrame(results_dict).T
    print(df)
    df.plot(kind='bar', figsize=(10, 5), ylim=(0,1))
    plt.title("Porovnanie Precision / Recall / F1")
    plt.ylabel("Score")
    plt.show()

def plot_training_from_custom_results(csv_path):
    df = pd.read_csv(csv_path)

    # Nastav dizajn
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Tréningové straty
    axs[0].plot(df["epoch"], df["train/box_loss"], label="Box Loss")
    axs[0].plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
    axs[0].plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
    axs[0].set_title("Training Losses")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Metriky
    axs[1].plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    axs[1].plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    axs[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
    axs[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
    axs[1].set_title("Validation Metrics")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Metric")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# ======= Vyhodnocovanie =======
def run_evaluation(detector_name, detect_fn):
    total_TP, total_FP, total_FN = 0, 0, 0

    for fname in tqdm(os.listdir(image_dir)):
        if not fname.endswith(".jpg"):
            continue

        img_path = os.path.join(image_dir, fname)
        label_path = os.path.join(annotation_dir, fname.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        h, w = img.shape[:2]


        # Predspracovanie
        if USE_PREPROCESSING:
            img = preprocess_image(img)

        # Načítaj anotácie
        with open(label_path, "r") as f:
            gt_boxes = [yolo_to_bbox(line, w, h) for line in f.readlines()]

        # Detekcia
        pred_boxes = detect_fn(img)

        # Vyhodnoť
        TP, FP, FN = evaluate_detections(pred_boxes, gt_boxes)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    precision = total_TP / (total_TP + total_FP + 1e-6)
    recall = total_TP / (total_TP + total_FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    print(f"\n📊 Výsledky pre {detector_name} (predspracovanie: {USE_PREPROCESSING})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

# ======= Definície detektorov =======ň
def haar_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)
    return [[x, y, x + w, y + h] for (x, y, w, h) in faces]

def yolo_detector(img):
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   # YOLO očakáva obraz v RGB formáte
   results = yolo_model.predict(img_rgb, device="cpu", verbose=False)

    # Extrahuj bounding boxy
   pred_boxes = []
   for result in results:
       for box in result.boxes.xyxy:
           x1, y1, x2, y2 = map(int, box.tolist())
           pred_boxes.append([x1, y1, x2, y2])

   return pred_boxes

def custom_yolo_detector(img):
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   # YOLO očakáva obraz v RGB formáte
   results = custom_yolo_model.predict(img_rgb, device="cpu", verbose=False)

    # Extrahuj bounding boxy
   pred_boxes = []
   for result in results:
       for box in result.boxes.xyxy:
           x1, y1, x2, y2 = map(int, box.tolist())
           pred_boxes.append([x1, y1, x2, y2])

   return pred_boxes

def default_yolo_detector(img):
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   # YOLO očakáva obraz v RGB formáte
   results = default_yolo_model.predict(img_rgb, device="cpu", verbose=False)

    # Extrahuj bounding boxy
   pred_boxes = []
   for result in results:
       for box in result.boxes.xyxy:
           x1, y1, x2, y2 = map(int, box.tolist())
           pred_boxes.append([x1, y1, x2, y2])

   return pred_boxes

def train_custom_yolo(data_yaml, model_output='runs/train/faces'):
    model = YOLO('yolov8n.pt')  # Môžeš použiť aj 'yolov8s.pt'
    results = model.train(
        data=data_yaml,
        epochs=1,
        imgsz=640,
        batch=16,
        project=model_output,
        name='custom',
        device='cpu'  # alebo 'privateuseone:0' pre DirectML
    )
    return model


# def yolo_detector(img):
#     # Resize a transformuj na vstup YOLOv8
#     img_resized = cv2.resize(img, (640, 640))
#     img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
#     img_input = img_rgb.astype(np.float32) / 255.0
#     img_input = np.transpose(img_input, (2, 0, 1))  # (HWC → CHW)
#     img_input = np.expand_dims(img_input, axis=0)   # (1, 3, 640, 640)
#
#     inputs = {onnx_session.get_inputs()[0].name: img_input}
#     outputs = onnx_session.run(None, inputs)
#
#     output = outputs[0][0]  # tvar (num_boxes, 85) – xywh + conf + classes
#
#     pred_boxes = []
#     conf_threshold = 0.3
#     for det in output:
#         conf = det[4]
#         if conf > conf_threshold:
#             x, y, w, h = det[:4]
#             x1 = int((x - w / 2) * img.shape[1] / 640)
#             y1 = int((y - h / 2) * img.shape[0] / 640)
#             x2 = int((x + w / 2) * img.shape[1] / 640)
#             y2 = int((y + h / 2) * img.shape[0] / 640)
#             pred_boxes.append([x1, y1, x2, y2])
#
#     return pred_boxes


picture_collage(image_dir, annotation_dir)

# ======= Rozdelenie datasetu =======

split_dataset(image_dir, annotation_dir)

# ======= Trénovanie vlastného modelu =======

model = train_custom_yolo("yolo-faces.yaml")

# ======= Spusti vyhodnotenie =======

run_evaluation("Haar Cascade", haar_detector)
run_evaluation("YOLOv8", yolo_detector)
run_evaluation("Custom YOLOv8", custom_yolo_detector)
run_evaluation("Default YOLOv8", default_yolo_detector)
evaluate_model(model, "datasets/data_split/test/images", "datasets/data_split/test/labels")
plot_training_from_custom_results("runs/train/faces/custom2/results.csv")