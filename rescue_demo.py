import argparse
import csv
from pathlib import Path
import cv2

THRESHOLD = 200
MIN_AREA = 30
MAX_AREA = 1000
MIN_RATIO = 1.2


def detect_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, mask = cv2.threshold(blurred, THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h / max(w, 1) < MIN_RATIO:
            continue
        boxes.append((x, y, w, h))
    return sorted(boxes)


def draw_frame(frame, filename, index, total, boxes, scale=2):
    display = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = display.shape[:2]
    top = 74
    bottom = 42
    canvas = cv2.copyMakeBorder(display, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(22, 25, 28))
    cv2.putText(canvas, 'THE AVENGERS | THERMAL RESCUE PROTOTYPE', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, f'Frame {index}/{total}  |  {filename}', (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (195, 195, 195), 1, cv2.LINE_AA)
    for i, (x, y, bw, bh) in enumerate(boxes, 1):
        cv2.rectangle(canvas, (x * scale, y * scale + top), ((x + bw) * scale, (y + bh) * scale + top), (0, 220, 80), 2)
        cv2.putText(canvas, str(i), (x * scale, max(top + 14, y * scale + top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
    status = f'REVIEW REQUIRED: {len(boxes)} candidate region(s)' if boxes else 'NO CANDIDATE REGIONS'
    cv2.putText(canvas, status, (12, top + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255) if boxes else (180, 220, 180), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description='Thermal candidate detection live demo and MP4 export')
    parser.add_argument('--mode', choices=['live', 'video', 'both'], default='both')
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--input', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('rescue_demo_output'))
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    frames = sorted(args.input.glob('img_*.bmp'))
    if not frames:
        parser.error(f'No img_*.bmp files found in {args.input.resolve()}')
    args.output.mkdir(parents=True, exist_ok=True)
    writer = None
    video_target = None
    rows = []
    shown = 0
    try:
        for index, path in enumerate(frames, 1):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f'WARNING: Cannot read {path}; skipping')
                continue
            boxes = detect_candidates(frame)
            annotated = draw_frame(frame, path.name, index, len(frames), boxes)
            if args.mode in ('video', 'both'):
                if writer is None:
                    height, width = annotated.shape[:2]
                    # Some macOS OpenCV builds cannot encode mp4v. Try
                    # common MP4 encoders first, then Motion-JPEG AVI.
                    attempts = [
                        ('thermal_rescue_demo.mp4', 'mp4v'),
                        ('thermal_rescue_demo.mp4', 'avc1'),
                        ('thermal_rescue_demo.avi', 'MJPG'),
                        ('thermal_rescue_demo.avi', 'XVID'),
                    ]
                    for filename, codec in attempts:
                        target = args.output / filename
                        candidate = cv2.VideoWriter(
                            str(target), cv2.VideoWriter_fourcc(*codec),
                            args.fps, (width, height), True
                        )
                        if candidate.isOpened():
                            writer = candidate
                            video_target = target
                            print(f'Video writer ready: {target} (codec {codec})')
                            break
                        candidate.release()
                        if target.exists() and target.stat().st_size == 0:
                            target.unlink()
                    if writer is None:
                        raise RuntimeError(
                            'No supported video encoder found (mp4v, avc1, MJPG, XVID). '
                            'Run --mode live or install ffmpeg for conversion.'
                        )
                if annotated.shape[:2] != (height, width):
                    annotated = cv2.resize(annotated, (width, height))
                writer.write(annotated)
            for candidate_index, (x, y, w, h) in enumerate(boxes, 1):
                rows.append([path.name, candidate_index, x, y, w, h, 'OPERATOR_REVIEW'])
            if not boxes:
                rows.append([path.name, 0, '', '', '', '', 'NO_CANDIDATE'])
            print(f'{path.name}: {len(boxes)} candidate(s) | simulated operator review only')
            shown += 1
            if args.mode in ('live', 'both'):
                try:
                    cv2.imshow('Thermal Rescue Demo - press Q to stop', annotated)
                    if cv2.waitKey(max(1, round(1000 / args.fps))) & 0xFF == ord('q'):
                        print('Live preview stopped by user.')
                        break
                except cv2.error as error:
                    raise RuntimeError('Live window failed. Try --mode video on a headless system.') from error
    finally:
        if writer is not None:
            writer.release()
        if args.mode in ('live', 'both'):
            cv2.destroyAllWindows()
        with (args.output / 'detection_log.csv').open('w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['frame', 'candidate_id', 'x', 'y', 'width', 'height', 'status'])
            csv_writer.writerows(rows)
    print(f'Processed {shown} frame(s). Outputs in {args.output.resolve()}')
    if args.mode in ('video', 'both'):
        if video_target is not None:
            print(f'Video saved: {video_target.resolve()}')
    print('Log: detection_log.csv')
    print('DISCLAIMER: Candidate warm regions are not verified people; no GPS or real rescue alerts.')


if __name__ == '__main__':
    main()
