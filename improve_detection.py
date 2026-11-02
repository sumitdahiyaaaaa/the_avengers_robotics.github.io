
import cv2
import csv
import re
from pathlib import Path

# ==========================================
# THERMAL SEARCH AND RESCUE DRONE
# Team 3 - The Avengers
# Detection Parameter Experiments
# ==========================================

DATASET = Path(".")
GROUND_TRUTH = "groundTruth.txt"
IOU_THRESHOLD = 0.3

# Previous baseline results
BASELINE_PRECISION = 74 / (74 + 643)
BASELINE_RECALL = 74 / (74 + 17)

# Try several detection configurations
CONFIGS = [
    (160, 10, 10000, 0.0),
    (180, 20, 10000, 0.0),
    (180, 30, 2000, 0.5),
    (180, 50, 1500, 0.8),
    (190, 20, 2000, 0.5),
    (190, 30, 1500, 0.8),
    (200, 20, 1200, 1.0),
    (200, 30, 1000, 1.2),
]


def load_annotations():
    annotations = {}

    with open(GROUND_TRUTH) as file:
        for line in file:
            if not line.startswith("img_"):
                continue

            name = line.split()[0]

            matches = re.findall(
                r"\(\s*(\d+)\s+(\d+)\s+"
                r"(\d+)\s+(\d+)\s*\)",
                line
            )

            annotations[name] = [
                tuple(map(int, m))
                for m in matches
            ]

    return annotations


def iou(a, b):
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


def detect(image, config):
    threshold, min_area, max_area, min_ratio = config

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    enhanced = cv2.equalizeHist(gray)

    blurred = cv2.GaussianBlur(
        enhanced, (5, 5), 0
    )

    _, mask = cv2.threshold(
        blurred, threshold, 255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    predictions = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        ratio = h / max(w, 1)

        if ratio < min_ratio:
            continue

        predictions.append(
            (x, y, x + w, y + h)
        )

    return predictions


def evaluate(predictions, labels):
    possible = []

    for i, pred in enumerate(predictions):
        for j, label in enumerate(labels):
            score = iou(pred, label)

            if score >= IOU_THRESHOLD:
                possible.append((score, i, j))

    possible.sort(reverse=True)

    used_predictions = set()
    used_labels = set()

    for score, i, j in possible:
        if i in used_predictions or j in used_labels:
            continue

        used_predictions.add(i)
        used_labels.add(j)

    tp = len(used_predictions)
    fp = len(predictions) - tp
    fn = len(labels) - tp

    return tp, fp, fn


def main():
    annotations = load_annotations()

    # Use first 15 images for development.
    # Reserve remaining 16 for later evaluation.
    names = sorted(annotations)
    development = names[:15]
    held_out = names[15:]

    print(f"Development frames: {len(development)}")
    print(f"Held-out frames: {len(held_out)}")

    # Load images once
    images = {}

    for name in names:
        image = cv2.imread(str(DATASET / name))

        if image is None:
            raise FileNotFoundError(name)

        images[name] = image

    results = []

    for config in CONFIGS:
        tp = fp = fn = 0

        for name in development:
            predictions = detect(images[name], config)
            labels = annotations[name]

            a, b, c = evaluate(predictions, labels)

            tp += a
            fp += b
            fn += c

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        results.append({
            "config": config,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        print(
            f"Config {config}: "
            f"P={precision:.2%}, "
            f"R={recall:.2%}, "
            f"F1={f1:.2%}"
        )

    # Choose highest development F1 score
    best = max(results, key=lambda r: r["f1"])

    print("\n===== BEST DEVELOPMENT CONFIG =====")
    print(f"Settings: {best['config']}")
    print(f"Precision: {best['precision']:.2%}")
    print(f"Recall: {best['recall']:.2%}")
    print(f"F1: {best['f1']:.2%}")

    # Evaluate selected configuration on held-out frames
    tp = fp = fn = 0

    for name in held_out:
        predictions = detect(images[name], best["config"])

        a, b, c = evaluate(
            predictions, annotations[name]
        )

        tp += a
        fp += b
        fn += c

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("\n===== HELD-OUT TEST RESULTS =====")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 score: {f1:.2%}")

    with open("experiment_results.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Threshold", "Min Area", "Max Area",
            "Min Ratio", "TP", "FP", "FN",
            "Precision", "Recall", "F1"
        ])

        for result in results:
            writer.writerow([
                *result["config"],
                result["tp"],
                result["fp"],
                result["fn"],
                round(result["precision"], 4),
                round(result["recall"], 4),
                round(result["f1"], 4)
            ])

    print("\nSaved experiment_results.csv")


if __name__ == "__main__":
    main()
