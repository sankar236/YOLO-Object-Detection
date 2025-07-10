import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import os
import sqlite3
from datetime import datetime
from multiprocessing import Process
from sklearn.metrics.pairwise import cosine_distances

# ---------- Setup ----------
model_path = "final.pt"
face_db_path = "./faces"
os.makedirs("violations", exist_ok=True)
db_path = "violations.db"
threshold_iou = 0.3
deepface_threshold = 0.65
cooldown_frames = 100

# ---------- Database ----------
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        timestamp TEXT,
        image_path TEXT,
        cam_id TEXT
    )''')
    conn.commit()
    conn.close()

def log_violation(name, timestamp, image_path, cam_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO violations (name, timestamp, image_path, cam_id) VALUES (?, ?, ?, ?)",
                   (name, timestamp, image_path, cam_id))
    conn.commit()
    conn.close()

init_db()

# ---------- Enhancements ----------
def enhance_low_light_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    gamma = 1.4
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    final = cv2.LUT(enhanced, table)
    return final

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * (yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# ---------- Face Embedding Loader ----------
def load_face_db():
    db_embeddings = []
    db_labels = []
    for root, _, files in os.walk(face_db_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                try:
                    emb = DeepFace.represent(img_path=full_path, model_name="VGG-Face", enforce_detection=True)[0]["embedding"]
                    db_embeddings.append(emb)
                    db_labels.append(os.path.splitext(file)[0])
                except Exception as e:
                    print(f"[ERROR] Failed to process {file}: {e}")
    return np.array(db_embeddings), db_labels

# ---------- Camera Processor ----------
def process_camera(cam_id):
    cap = cv2.VideoCapture(cam_id)
    yolo_model = YOLO(model_path)
    cooldown_tracker = {}
    face_name_map = {}
    db_embeddings, db_labels = load_face_db()

    frame_count = 0
    while cap.isOpened():
        ret, frame_original = cap.read()
        if not ret:
            break

        frame = frame_original.copy()
        enhanced_frame = enhance_low_light_image(frame)
        frame_count += 1

        yolo_results = yolo_model(enhanced_frame)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
        names = yolo_model.model.names

        helmet_boxes = [box for box, cls in zip(boxes, class_ids) if names[cls] == "helmet"]
        head_boxes = [box for box, cls in zip(boxes, class_ids) if names[cls] == "head"]

        for box in head_boxes:
            x1, y1, x2, y2 = map(int, box)
            face_bbox = [x1, y1, x2, y2]
            has_helmet = any(compute_iou(face_bbox, hb) > threshold_iou for hb in helmet_boxes)

            face_id = f"{x1}-{y1}-{x2}-{y2}"
            person_name = face_name_map.get(face_id, None)

            if has_helmet:
                label = person_name if person_name else "helmet"
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(enhanced_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            else:
                face_roi = frame[y1:y2, x1:x2]

                if face_roi.shape[0] < 80 or face_roi.shape[1] < 80:
                    continue

                cv2.imshow("Face ROI", face_roi)
                cv2.waitKey(1)

                last_logged = cooldown_tracker.get(face_id, -cooldown_frames)
                if frame_count - last_logged < cooldown_frames:
                    continue

                try:
                    emb_result = DeepFace.represent(face_roi, model_name="VGG-Face", enforce_detection=False)
                    if len(emb_result) == 0:
                        raise ValueError("No face found")
                    emb = emb_result[0]["embedding"]
                    distances = cosine_distances([emb], db_embeddings)[0]
                    best_idx = np.argmin(distances)
                    best_distance = distances[best_idx]

                    if best_distance <= deepface_threshold:
                        person_name = db_labels[best_idx]
                        print(f"[DEBUG] Match: {person_name}, Distance: {best_distance:.4f}")
                    else:
                        print(f"[DEBUG] No match (min distance: {best_distance:.4f})")
                        person_name = "unknown"
                except Exception as e:
                    print(f"[ERROR] Embedding failed: {e}")
                    person_name = "unknown"

                face_name_map[face_id] = person_name

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                snapshot_name = f"{person_name}_{timestamp}_{cam_id}.jpg"
                snapshot_path = os.path.join("violations", snapshot_name).replace("\\", "/")
                cv2.imwrite(snapshot_path, face_roi)
                log_violation(person_name, timestamp, snapshot_path, str(cam_id))
                cooldown_tracker[face_id] = frame_count

                print(f"[VIOLATION][Cam {cam_id}] {person_name} at {timestamp}")
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(enhanced_frame, f"no helmet: {person_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        for box in helmet_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        cv2.imshow(f"Enhanced View - Cam {cam_id}", enhanced_frame)
        cv2.imshow(f"Original View - Cam {cam_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- Main ----------
if __name__ == "__main__":
    camera_ids = [0]
    processes = [Process(target=process_camera, args=(cam_id,)) for cam_id in camera_ids]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
