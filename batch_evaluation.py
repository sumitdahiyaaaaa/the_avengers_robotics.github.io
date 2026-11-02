
import cv2
import csv
import re
from pathlib import Path

# ==========================================
# THERMAL SEARCH AND RESCUE DRONE
# Team 3 - The Avengers
# Batch Evaluation - 31 Thermal Images
# ==========================================

THRESHOLD = 180
MIN_AREA = 20
IOU_THRESHOLD = 0.3

DATASET_PATH = Path(".")
OUTPUT_FOLDER = Path("batch_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)


def load_annotations():
    annotations = {}

    with open("groundTruth.txt", "r") as file:
        for line in file:
            line = line.strip()

            if not line.startswith("img_"):
                continue

            filename = line.split()[0]

            matches = re.findall(
                r"\(\s*(\d+)\s+(\d+)\s+"
                r"(\d+)\s+(\d+)\s*\)",
                line
            )

            annotations[filename] = [
                tuple(map(int, match))
                for match in matches
            ]

    return annotations


def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area_a = (
        (a[2] - a[0]) *
        (a[3] - a[1])
    )

    area_b = (
        (b[2] - b[0]) *
        (b[3] - b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def detect_people(image):
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
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    predictions = []

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        predictions.append(
            (x, y, x + w, y + h)
        )

    return predictions


def evaluate(predictions, labels):
    matches = []

    for i, prediction in enumerate(predictions):
        for j, label in enumerate(labels):
            iou = calculate_iou(prediction, label)

            if iou >= IOU_THRESHOLD:
                matches.append((iou, i, j))

    matches.sort(reverse=True)

    matched_predictions = set()
    matched_labels = set()

    for iou, i, j in matches:
        if i in matched_predictions:
            continue

        if j in matched_labels:
            continue

        matched_predictions.add(i)
        matched_labels.add(j)

    tp = len(matched_predictions)
    fp = len(predictions) - tp
    fn = len(labels) - tp

    return tp, fp, fn


def main():
    annotations = load_annotations()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    results = []

    for filename in sorted(annotations):
        image_path = DATASET_PATH / filename
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping missing image: {filename}")
            continue

        labels = annotations[filename]
        predictions = detect_people(image)

        tp, fp, fn = evaluate(predictions, labels)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = (
            tp / (tp + fp)
            if tp + fp > 0 else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0 else 0
        )

        results.append([
            filename,
            len(labels),
            len(predictions),
            tp, fp, fn,
            round(precision, 4),
            round(recall, 4)
        ])

        # Draw ground-truth boxes in blue
        for x1, y1, x2, y2 in labels:
            cv2.rectangle(
                image,
                (x1, y1), (x2, y2),
                (255, 0, 0), 1
            )

        # Draw predicted boxes in green
        for x1, y1, x2, y2 in predictions:
            cv2.rectangle(
                image,
                (x1, y1), (x2, y2),
                (0, 255, 0), 1
            )

        cv2.imwrite(
            str(OUTPUT_FOLDER / f"{Path(filename).stem}.png"),
            image
        )

        print(
            f"{filename}: TP={tp}, FP={fp}, FN={fn}"
        )

    overall_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0
    )

    overall_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0 else 0
    )

    with open("batch_report.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Image", "Labeled People",
            "Predictions", "TP", "FP", "FN",
            "Precision", "Recall"
        ])

        writer.writerows(results)

    print("\n===== FINAL BATCH REPORT =====")
    print(f"Images evaluated: {len(results)}")
    print(f"True positives: {total_tp}")
    print(f"False positives: {total_fp}")
    print(f"False negatives: {total_fn}")
    print(f"Overall precision: {overall_precision:.2%}")
    print(f"Overall recall: {overall_recall:.2%}")
    print("Report saved: batch_report.csv")
    print("Annotated images saved: batch_results/")


if __name__ == "__main__":
    main()
