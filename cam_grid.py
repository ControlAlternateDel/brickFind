import math
import os
import time

import cv2


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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("Press 'm' to save the marked divisions, 'q' or ESC to quit.")
    os.makedirs("captures", exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

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
            for r in range(n):
                for c in range(n):
                    if idx >= divisions:
                        break
                    x0, y0 = w * c // n, h * r // n
                    x1, y1 = w * (c + 1) // n, h * (r + 1) // n
                    cv2.imwrite(f"captures/{stamp}-r{r}-c{c}.jpg",
                                frame[y0:y1, x0:x1])
                    idx += 1
            print(f"Saved {idx} images to captures/")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
