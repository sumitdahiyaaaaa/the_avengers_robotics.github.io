
import cv2
import csv
import re
from pathlib import Path

# ==========================================
# THERMAL SEARCH AND RESCUE DRONE
# Team 3 - The Avengers
# Ground Truth Evaluation
# ==========================================

IMAGE_NAME = "img_00001.bmp"
GROUND_TRUTH_FILE = "groundTruth.txt"
OUTPUT_IMAGE = "evaluation_result.png"
OUTPUT_CSV = "evaluation_report.csv"

# Detection settings
THRESHOLD = 180
MIN_AREA = 20
MIN_IOU = 0.3


def load_ground_truth(filename, image_name):
    """Read pedestrian boxes from OSU annotations."""
    with open(filename, "r") as file:
        lines = file.readlines()

    for line in lines:
        if not line.startswith(image_name + " "):
            continue

        count = int(line.split()[1])

        matches = re.findall(
            r"\(\s*(\d+)\s+(\d+)\s+"
            r"(\d+)\s+(\d+)\s*\)",
            line
        )

        boxes = [
            tuple(map(int, match))
            for match in matches
        ]

        if len(boxes) != count:
            raise ValueError(
                "Ground-truth box count mismatch"
            )

        return boxes

    raise ValueError(
        f"No annotations found for {image_name}"
    )


def calculate_iou(box_a, box_b):
    """Intersection over Union of two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area_a = (
        (box_a[2] - box_a[0]) *
        (box_a[3] - box_a[1])
    )

    area_b = (
        (box_b[2] - box_b[0]) *
        (box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def detect_regions(image):
    """Find bright thermal regions."""
    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    enhanced = cv2.equalizeHist(gray)

    blurred = cv2.GaussianBlur(
        enhanced, (5, 5), 0
    )

    _, mask = cv2.threshold(
        blurred, THRESHOLD, 255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        boxes.append((x, y, x + w, y + h))

    return boxes


def evaluate(predictions, ground_truth):
    """Match predictions to labels one-to-one."""
    possible_matches = []

    for pred_id, pred in enumerate(predictions):
        for gt_id, gt in enumerate(ground_truth):
            iou = calculate_iou(pred, gt)

            if iou >= MIN_IOU:
                possible_matches.append(
                    (iou, pred_id, gt_id)
                )

    possible_matches.sort(reverse=True)

    matched_predictions = set()
    matched_ground_truth = set()

    for iou, pred_id, gt_id in possible_matches:
        if pred_id in matched_predictions:
            continue

        if gt_id in matched_ground_truth:
            continue

        matched_predictions.add(pred_id)
        matched_ground_truth.add(gt_id)

    tp = len(matched_predictions)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp

    return tp, fp, fn


def main():
    if not Path(IMAGE_NAME).exists():
        raise FileNotFoundError(IMAGE_NAME)

    image = cv2.imread(IMAGE_NAME)

    if image is None:
        raise ValueError("Could not load image")

    ground_truth = load_ground_truth(
        GROUND_TRUTH_FILE, IMAGE_NAME
    )

    predictions = detect_regions(image)

    tp, fp, fn = evaluate(
        predictions, ground_truth
    )

    # Draw labeled pedestrian boxes in blue
    for box in ground_truth:
        cv2.rectangle(
            image,
            (box[0], box[1]),
            (box[2], box[3]),
            (255, 0, 0), 1
        )

    # Draw predicted regions in green
    for box in predictions:
        cv2.rectangle(
            image,
            (box[0], box[1]),
            (box[2], box[3]),
            (0, 255, 0), 1
        )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0
    )

    cv2.imwrite(OUTPUT_IMAGE, image)

    with open(OUTPUT_CSV, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Image", "Ground Truth",
            "Predictions", "TP", "FP", "FN",
            "Precision", "Recall"
        ])

        writer.writerow([
            IMAGE_NAME, len(ground_truth),
            len(predictions), tp, fp, fn,
            round(precision, 3),
            round(recall, 3)
        ])

    print("===== THERMAL DETECTION REPORT =====")
    print(f"Image: {IMAGE_NAME}")
    print(f"Labeled pedestrians: {len(ground_truth)}")
    print(f"Predicted regions: {len(predictions)}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"Missed pedestrians: {fn}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"Saved image: {OUTPUT_IMAGE}")
    print(f"Saved report: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
