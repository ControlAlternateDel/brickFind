import math
import os
import sqlite3
import time
import traceback

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
PIECES_PATH = os.path.join(BASE_DIR, "pieces.txt")
DB_PATH = os.path.join(BASE_DIR, "brickfind.db")

try:
    import numpy as np
    from tf_keras.models import load_model
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

'''
def add_pieces_to_db(pieces):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS Your_parts ("
            "name TEXT PRIMARY KEY, quantity INTEGER NOT NULL DEFAULT 1)"
        )
        for piece in pieces:
            conn.execute(
                "INSERT INTO Your_parts (name, quantity) VALUES (?, 1) "
                "ON CONFLICT(name) DO UPDATE SET quantity = quantity + 1",
                (piece,),
            )
        conn.commit()
    finally:
        conn.close()
'''

def load_piece_model():
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        labels = []
        for line in f:
            parts = line.split()
            if not parts:
                continue
            labels.append(parts[1] if len(parts) > 1 else parts[0])
    return model, labels


def classify_images(model, labels, paths):
    pieces = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"Could not read {p}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        probs = model.predict(img, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = labels[idx] if idx < len(labels) else f"class{idx}"
        print(f"{os.path.basename(p)} -> {label} ({probs[idx]:.2f})")
        pieces.append(label)
    return pieces


def main():
    divisions = 3
    while True:
        value = input(f"Number of grid divisions [{divisions}]: ").strip()
        if not value:
            break
        try:
            divisions = int(value)
            if divisions > 0:
                break
            print("Must be a positive integer.")
        except ValueError:
            print("Invalid number.")

    n = max(1, math.ceil(math.sqrt(divisions)))
    total = n * n
    if total != divisions:
        print(f"Rounding up to a {n}x{n} grid ({total} frames) to keep the "
              f"same aspect ratio as the full frame.")

    camnum = 0
    value = input(f"Enter camera number [{camnum}]: ").strip()
    if value:
        try:
            camnum = int(value)
        except ValueError:
            print(f"Invalid number, using camera {camnum}.")

    cap = cv2.VideoCapture(camnum, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera {camnum}.")
        return

    model = None
    labels = []
    if _HAS_ML:
        try:
            model, labels = load_piece_model()
            print(f"Loaded model with {len(labels)} classes: {', '.join(labels)}")
        except Exception as exc:
            print(f"Could not load model, skipping classification: {exc}")
    else:
        print("keras/numpy not installed; images will be saved without classification.")

    pieces = []
    print("Press 'm' to save the marked divisions, 'q' or ESC to quit.")
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        clean = frame.copy()
        h, w = frame.shape[:2]
        for i in range(1, n):
            x = w * i // n
            y = h * i // n
            cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 1)
            cv2.line(frame, (0, y), (w, y), (0, 255, 0), 1)

        idx = 0
        for r in range(n):
            for c in range(n):
                x0, y0 = w * c // n, h * r // n
                x1, y1 = w * (c + 1) // n, h * (r + 1) // n
                if idx < divisions:
                    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1),
                                  (0, 255, 0), 2)
                else:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1),
                                  (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1),
                                  (0, 0, 255), 1)
                idx += 1

        cv2.imshow("Grid", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            idx = 0
            saved = []
            for r in range(n):
                for c in range(n):
                    if idx >= divisions:
                        break
                    x0, y0 = w * c // n, h * r // n
                    x1, y1 = w * (c + 1) // n, h * (r + 1) // n
                    path = os.path.join(CAPTURES_DIR, f"{stamp}-r{r}-c{c}.jpg")
                    cv2.imwrite(path, clean[y0:y1, x0:x1])
                    saved.append(path)
                    idx += 1
            print(f"Saved {idx} images to captures/")

            new_pieces = []
            try:
                if model is not None:
                    new_pieces = classify_images(model, labels, saved)
                    pieces.extend(new_pieces)
                else:
                    print("Model not loaded; skipping classification.")
            except Exception as exc:
                traceback.print_exc()
                print(f"Classification failed: {exc}")

            if new_pieces:
                """add_pieces_to_db(new_pieces)"""
                print(f"Added to database (table Your_parts): {new_pieces}")

            with open(PIECES_PATH, "w") as f:
                f.write("\n".join(pieces))
                f.write("\n")
            print(f"pieces.txt updated ({len(pieces)} pieces so far): {pieces}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
