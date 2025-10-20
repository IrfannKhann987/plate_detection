import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque

model = YOLO(r"I:\onr\new_onr\new_onr\license_plate_best.pt")
reader = easyocr.Reader(['en'], gpu=True)

# Example: GX15OGJ
plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")

def correct_plate_format(ocr_text):
    # Minimal correction mapping (only for obvious misreads)
    mapping_num_to_alpha = {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}
    mapping_alpha_to_num = {'O': '0', 'I': '1', 'S': '5', 'B': '8'}

    ocr_text = ocr_text.upper().replace(" ", "")
    if len(ocr_text) != 7:
        return ocr_text  # Don’t reject — just return raw OCR if close

    corrected = []
    for i, ch in enumerate(ocr_text):
        # UK-style pattern: LLDDLLL → first 2 letters, 2 digits, then 3 letters
        if i < 2 or i >= 4:  # Letter positions
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])
            else:
                corrected.append(ch)
        else:  # Number positions
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            else:
                corrected.append(ch)
    return "".join(corrected)


def recognize_plate(plate_crop):
    if plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(threshold, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        ocr_result = reader.readtext(
            plate_resized, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        if len(ocr_result) > 0:
            raw_text = ocr_result[0].upper()
            candidate = correct_plate_format(raw_text)
            if plate_pattern.match(candidate):
                return candidate
            else:
                return raw_text  # fallback to OCR text even if pattern doesn't match
    except:
        pass
    return ""


plate_history = defaultdict(lambda: deque(maxlen=10))
final_plate = {}

def get_box_id(x1, y1, x2, y2):
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"

def stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)
        most_common = max(set(plate_history[box_id]), key=plate_history[box_id].count)
        final_plate[box_id] = most_common
    return final_plate.get(box_id, "")


input_video = r"I:\onr\new_onr\new_onr\data\License Plate Detection Test (1).mp4"
output_video = "whatever_the_fuck_is.mp4"

cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(3)), int(cap.get(4))))

CONF_THRESH = 0.4

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf.cpu().numpy())
            if conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            plate_crop = frame[y1:y2, x1:x2]

            text = recognize_plate(plate_crop)
            box_id = get_box_id(x1, y1, x2, y2)
            stable_text = stable_plate(box_id, text)

            # Draw the rectangle *first*
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)

            # Draw text in a semi-transparent label box above the plate
            if stable_text:
                text_y = max(y1 - 10, 25)
                (tw, th), _ = cv2.getTextSize(stable_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(frame, (x1, text_y - th - 5), (x1 + tw + 10, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, stable_text, (x1 + 5, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Annotated_Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
