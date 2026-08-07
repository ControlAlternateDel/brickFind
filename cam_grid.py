import math
import os
import sqlite3
import time
import traceback

import customtkinter as ctk
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


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    cap.release()
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()
    return None


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


def draw_grid(frame, n, divisions):
    display = frame.copy()
    h, w = display.shape[:2]
    for i in range(1, n):
        x = w * i // n
        y = h * i // n
        cv2.line(display, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.line(display, (0, y), (w, y), (0, 255, 0), 1)

    idx = 0
    for r in range(n):
        for c in range(n):
            x0, y0 = w * c // n, h * r // n
            x1, y1 = w * (c + 1) // n, h * (r + 1) // n
            if idx < divisions:
                cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1),
                              (0, 255, 0), 2)
            else:
                overlay = display.copy()
                cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1),
                              (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
                cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1),
                              (0, 0, 255), 1)
            idx += 1
    return display


def capture_divisions(clean, n, divisions):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    h, w = clean.shape[:2]
    saved = []
    idx = 0
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
    return saved


class App(ctk.CTk):
    def __init__(self, divisions, model, labels):
        super().__init__()
        self.title("brickFind")
        self.geometry("360x340")
        self.resizable(False, False)

        self.divisions = divisions
        self.n = max(1, math.ceil(math.sqrt(divisions)))
        self.model = model
        self.labels = labels
        self.pieces = []
        self.clean = None
        self.camera_index = None
        self.cap = None

        self.available = [i for i in range(10) if open_camera(i) is not None]

        self._build_ui()

        if self.available:
            self.switch_camera(self.available[0])
        else:
            self.status.configure(text="No camera found")

        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.after(30, self.process_frame)

    def _build_ui(self):
        ctk.CTkLabel(self, text="brickFind", font=("Arial", 22, "bold")).pack(pady=(16, 4))
        ctk.CTkLabel(self, text=f"Grid: {self.divisions} divisions ({self.n}x{self.n})").pack(pady=(0, 10))

        self.camera_menu = ctk.CTkOptionMenu(
            self, values=[f"Camera {i}" for i in self.available],
            command=self._on_menu_select)
        self.camera_menu.pack(pady=6, padx=24, fill="x")

        nav = ctk.CTkFrame(self)
        nav.pack(pady=6, padx=24, fill="x")
        ctk.CTkButton(nav, text="Previous", command=self.previous_camera).pack(
            side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkButton(nav, text="Next", command=self.next_camera).pack(
            side="left", expand=True, fill="x", padx=(6, 0))

        ctk.CTkButton(self, text="Capture Pieces", command=self.capture).pack(
            pady=8, padx=24, fill="x")
        ctk.CTkButton(self, text="Quit", fg_color="#b02e2e", hover_color="#8f2424",
                      command=self.quit_app).pack(pady=(4, 8), padx=24, fill="x")

        self.status = ctk.CTkLabel(self, text="", wraplength=320)
        self.status.pack(pady=(4, 12))

    def switch_camera(self, index):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cap = open_camera(index)
        if self.cap is None:
            self.status.configure(text=f"Could not open camera {index}")
            return
        self.camera_index = index
        self.camera_menu.set(f"Camera {index}")
        self.status.configure(text=f"Camera {index}")

    def _on_menu_select(self, value):
        self.switch_camera(int(value.split()[-1]))

    def previous_camera(self):
        self._step_camera(-1)

    def next_camera(self):
        self._step_camera(1)

    def _step_camera(self, direction):
        if not self.available:
            return
        try:
            i = self.available.index(self.camera_index)
        except ValueError:
            i = 0
        self.switch_camera(self.available[(i + direction) % len(self.available)])

    def process_frame(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                self.clean = frame.copy()
                cv2.imshow("Grid", draw_grid(frame, self.n, self.divisions))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    self.quit_app()
                    return
                elif key == ord("m"):
                    self.capture()
        self.after(30, self.process_frame)

    def capture(self):
        if self.clean is None:
            self.status.configure(text="No frame available yet")
            return
        saved = capture_divisions(self.clean, self.n, self.divisions)
        new_pieces = []
        try:
            if self.model is not None:
                new_pieces = classify_images(self.model, self.labels, saved)
                self.pieces.extend(new_pieces)
            else:
                print("Model not loaded; skipping classification.")
        except Exception as exc:
            traceback.print_exc()
            print(f"Classification failed: {exc}")

        if new_pieces:
            add_pieces_to_db(new_pieces)

        with open(PIECES_PATH, "w") as f:
            f.write("\n".join(self.pieces))
            f.write("\n")

        self.status.configure(
            text=f"Saved {len(saved)} images; identified {len(new_pieces)} piece(s): {new_pieces}")

    def quit_app(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        self.destroy()


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

    os.makedirs(CAPTURES_DIR, exist_ok=True)

    app = App(divisions, model, labels)
    app.mainloop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
