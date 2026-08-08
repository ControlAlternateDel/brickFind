import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager

import customtkinter as ctk
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
PIECES_PATH = os.path.join(BASE_DIR, "pieces.txt")
DB_PATH = os.path.join(BASE_DIR, "brickfind.db")
SETS_PATH = os.path.join(BASE_DIR, "sets.json")
CAMERA_NAMES_PATH = os.path.join(BASE_DIR, "camera_names.json")

DISPLAY_NAMES = {
    "3003": "2x2",
    "3001": "2x4",
    "3004": "1x2",
    "3010": "1x4",
}


def display_name(label):
    return DISPLAY_NAMES.get(label, label)

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


@contextmanager
def silent_stderr():
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def load_camera_name_overrides():
    """Return {index: name} from camera_names.json (editable by the user)."""
    try:
        with open(CAMERA_NAMES_PATH) as f:
            data = json.load(f)
        return {int(k): str(v) for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {}


def save_camera_name_overrides(mapping):
    with open(CAMERA_NAMES_PATH, "w") as f:
        json.dump({str(k): v for k, v in sorted(mapping.items())}, f, indent=2)


def _macos_device_resolutions():
    """Return [(device name, {(w, h), ...})] via AVFoundation, or None."""
    try:
        import CoreMedia
        from AVFoundation import AVCaptureDevice
    except ImportError:
        return None
    devices = AVCaptureDevice.devicesWithMediaType_("vide")
    result = []
    for d in devices:
        dims = set()
        for fmt in d.formats():
            dim = CoreMedia.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
            dims.add((int(dim.width), int(dim.height)))
        result.append((d.localizedName(), dims))
    return result


def get_camera_names():
    """Return a dict mapping camera index to a display name (best effort)."""
    openable = []
    resolutions = {}
    with silent_stderr():
        for i in range(10):
            cap = open_camera(i)
            if cap is not None:
                openable.append(i)
                resolutions[i] = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                cap.release()

    names = {}
    if hasattr(cv2, "CAP_PROP_DEVICE_DESCRIPTION"):
        for i in openable:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                name = cap.get(cv2.CAP_PROP_DEVICE_DESCRIPTION)
                names[i] = str(name).strip() or f"Camera {i}"
            cap.release()
        return names
    if sys.platform == "darwin":
        dev_res = _macos_device_resolutions()
        if dev_res is not None:
            unassigned = list(dev_res)
            remaining = list(openable)
            while remaining:
                progressed = False
                for i in list(remaining):
                    cands = [dev for dev in unassigned
                             if resolutions[i] in dev[1]]
                    if len(cands) == 1:
                        names[i] = cands[0][0]
                        unassigned.remove(cands[0])
                        remaining.remove(i)
                        progressed = True
                if not progressed:
                    for i, dev in zip(remaining, unassigned):
                        names[i] = dev[0]
                    break
        else:
            try:
                out = subprocess.run(
                    ["system_profiler", "SPCameraDataType", "-json"],
                    capture_output=True, text=True, timeout=15).stdout
                cameras = json.loads(out).get("SPCameraDataType", [])
                for idx, i in enumerate(openable):
                    if idx < len(cameras):
                        names[i] = cameras[idx].get("_name") or f"Camera {i}"
                    else:
                        names[i] = f"Camera {i}"
            except Exception:
                pass
    for i in openable:
        names.setdefault(i, f"Camera {i}")
    overrides = load_camera_name_overrides()
    for i in openable:
        if i in overrides:
            names[i] = overrides[i]
    if not os.path.exists(CAMERA_NAMES_PATH):
        save_camera_name_overrides(names)
    return names


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


def clear_inventory():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS Your_parts ("
                     "name TEXT PRIMARY KEY, quantity INTEGER NOT NULL DEFAULT 1)")
        conn.execute("DELETE FROM Your_parts")
        conn.commit()
    finally:
        conn.close()


def load_sets():
    """Return {set name: {piece: required qty}} from sets.json."""
    try:
        with open(SETS_PATH) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Could not load {SETS_PATH}: {exc}")
        return {}
    return {name: dict(pieces) for name, pieces in data.items()}


def load_inventory():
    """Return {raw model label: quantity} accumulated in the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT name, quantity FROM Your_parts").fetchall()
    finally:
        conn.close()
    return {name: qty for name, qty in rows}


def buildable_sets(sets, inventory):
    """Return the set names whose pieces are all available in inventory."""
    ready = []
    for name, pieces in sets.items():
        if all(inventory.get(piece, 0) >= qty for piece, qty in pieces.items()):
            ready.append(name)
    return ready


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
        self.geometry("880x760")
        self.resizable(False, False)

        self.divisions = divisions
        self.n = max(1, math.ceil(math.sqrt(divisions)))
        self.model = model
        self.labels = labels
        self.pieces = []
        self.clean = None
        self.camera_index = None
        self.cap = None
        self.video_w, self.video_h = 360, 270
        self.sets = load_sets()

        self.camera_names = get_camera_names()
        self.available = sorted(self.camera_names)

        self._build_ui()
        self._refresh_sets()
        self._refresh_inventory()

        if self.available:
            self.switch_camera(self.available[0])
        else:
            self.status.configure(text="No camera found")

        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.after(30, self.process_frame)

    def _build_ui(self):
        outer = ctk.CTkFrame(self)
        outer.pack(fill="both", expand=True)

        left = ctk.CTkFrame(outer, width=200)
        left.pack(side="left", fill="y", padx=(12, 6), pady=12)
        left.pack_propagate(False)
        ctk.CTkLabel(left, text="Your pieces:", font=("Arial", 13, "bold")).pack(
            anchor="w", padx=12, pady=(8, 2))
        self.inventory_panel = ctk.CTkScrollableFrame(left)
        self.inventory_panel.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        main = ctk.CTkFrame(outer)
        main.pack(side="left", fill="both", expand=True, padx=6)

        ctk.CTkLabel(main, text="brickFind", font=("Arial", 22, "bold")).pack(pady=(16, 4))
        self.grid_label = ctk.CTkLabel(
            main, text=f"Grid: {self.divisions} divisions ({self.n}x{self.n})")
        self.grid_label.pack(pady=(0, 10))

        div_row = ctk.CTkFrame(main)
        div_row.pack(pady=6, padx=24, fill="x")
        ctk.CTkLabel(div_row, text="Divisions:").pack(side="left", padx=(0, 6))
        self.div_entry = ctk.CTkEntry(div_row, width=60)
        self.div_entry.insert(0, str(self.divisions))
        self.div_entry.pack(side="left", padx=(0, 6))
        ctk.CTkButton(div_row, text="Apply", width=60,
                      command=self._apply_divisions).pack(side="left")
        self.div_entry.bind("<Return>", lambda _e: self._apply_divisions())

        self.camera_menu = ctk.CTkOptionMenu(
            main, values=[self._camera_label(i) for i in self.available],
            command=self._on_menu_select)
        self.camera_menu.pack(pady=6, padx=24, fill="x")

        nav = ctk.CTkFrame(main)
        nav.pack(pady=6, padx=24, fill="x")
        ctk.CTkButton(nav, text="Previous", command=self.previous_camera).pack(
            side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkButton(nav, text="Next", command=self.next_camera).pack(
            side="left", expand=True, fill="x", padx=(6, 0))

        self.video_label = ctk.CTkLabel(
            main, text="No camera found", fg_color="#1a1a1a",
            width=self.video_w, height=self.video_h)
        self.video_label.pack(pady=6, padx=24)

        ctk.CTkButton(main, text="Capture Pieces", command=self.capture).pack(
            pady=8, padx=24, fill="x")
        ctk.CTkButton(main, text="Clear Cached Inventory", fg_color="#8f5b2e",
                      hover_color="#774a24", command=self.clear_cached).pack(
            pady=(4, 8), padx=24, fill="x")
        ctk.CTkButton(main, text="Quit", fg_color="#b02e2e", hover_color="#8f2424",
                      command=self.quit_app).pack(pady=(4, 8), padx=24, fill="x")

        self.status = ctk.CTkLabel(main, text="", wraplength=360)
        self.status.pack(pady=(4, 12))

        right = ctk.CTkFrame(outer, width=200)
        right.pack(side="left", fill="y", padx=(6, 12), pady=12)
        right.pack_propagate(False)
        ctk.CTkLabel(right, text="Buildable sets:", font=("Arial", 13, "bold")).pack(
            anchor="w", padx=12, pady=(8, 2))
        self.sets_panel = ctk.CTkScrollableFrame(right)
        self.sets_panel.pack(fill="both", expand=True, padx=12, pady=(0, 8))

    def _camera_label(self, index):
        base = self.camera_names.get(index, f"Camera {index}")
        dup = [i for i in self.available
               if i != index and self.camera_names.get(i) == base]
        return f"{base} [{index}]" if dup else base

    def _apply_divisions(self):
        try:
            d = int(self.div_entry.get().strip())
            if d <= 0:
                self.status.configure(text="Divisions must be a positive integer")
                return
            self.divisions = d
            self.n = max(1, math.ceil(math.sqrt(d)))
            self.grid_label.configure(
                text=f"Grid: {self.divisions} divisions ({self.n}x{self.n})")
            self.status.configure(text=f"Grid set to {d} divisions")
        except ValueError:
            self.status.configure(text="Invalid number of divisions")

    def clear_cached(self):
        clear_inventory()
        self.pieces = []
        with open(PIECES_PATH, "w") as f:
            f.write("")
        self.status.configure(text="Cleared cached inventory")
        self._refresh_sets()
        self._refresh_inventory()

    def _refresh_sets(self):
        for widget in self.sets_panel.winfo_children():
            widget.destroy()
        if not self.sets:
            ctk.CTkLabel(self.sets_panel, text="No sets defined in sets.json",
                         text_color="gray", wraplength=160, justify="left").pack(
                anchor="w", padx=6, pady=1)
            return
        inventory = load_inventory()
        ready = buildable_sets(self.sets, inventory)
        for name in sorted(self.sets):
            missing = {piece: qty - inventory.get(piece, 0)
                       for piece, qty in self.sets[name].items()
                       if inventory.get(piece, 0) < qty}
            if missing:
                text = f"- {name}:\n    missing "
                text += ", ".join(
                    f"{display_name(p)} x{q}" for p, q in sorted(missing.items()))
                color = "gray"
            else:
                text = f"- {name}: READY"
                color = "#7ecf6a"
            ctk.CTkLabel(self.sets_panel, text=text, text_color=color,
                         wraplength=160, justify="left").pack(
                anchor="w", padx=6, pady=1)
        print(f"Sets you can make: {', '.join(ready) if ready else 'none'}")

    def _refresh_inventory(self):
        for widget in self.inventory_panel.winfo_children():
            widget.destroy()
        inventory = load_inventory()
        if not inventory:
            ctk.CTkLabel(self.inventory_panel, text="No pieces yet",
                         text_color="gray").pack(anchor="w", padx=6, pady=1)
            return
        for name in sorted(inventory):
            shown = display_name(name)
            ctk.CTkLabel(self.inventory_panel, text=f"{shown}: {inventory[name]}").pack(
                anchor="w", padx=6, pady=1)

    def switch_camera(self, index):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cap = open_camera(index)
        if self.cap is None:
            self.status.configure(text=f"Could not open {self._camera_label(index)}")
            return
        self.camera_index = index
        self.camera_menu.set(self._camera_label(index))
        self.status.configure(text=self._camera_label(index))
        self._fit_window_to_camera()

    def _fit_window_to_camera(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            return
        self.video_w = min(w, 360)
        self.video_h = round(self.video_w * h / w)
        self.video_label.configure(width=self.video_w, height=self.video_h)
        self.update_idletasks()
        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")

    def _on_menu_select(self, value):
        for i in self.available:
            if self._camera_label(i) == value:
                self.switch_camera(i)
                return

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
                self._update_video(frame)
        self.after(30, self.process_frame)

    def _update_video(self, frame):
        h, w = frame.shape[:2]
        scale = max(self.video_w / w, self.video_h / h)
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(frame, (nw, nh))
        x0 = (nw - self.video_w) // 2
        y0 = (nh - self.video_h) // 2
        img = img[y0:y0 + self.video_h, x0:x0 + self.video_w]
        img = draw_grid(img, self.n, self.divisions)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ctk_img = ctk.CTkImage(Image.fromarray(img), size=(self.video_w, self.video_h))
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img

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
        self._refresh_sets()
        self._refresh_inventory()

    def quit_app(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.destroy()


def main():
    divisions = 3

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


if __name__ == "__main__":
    main()
