import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("video/federal.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("output_speed_estimated_federal.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize SpeedEstimator
speedestimator = solutions.SpeedEstimator(
    model="yolo12s.pt",
    show=True,
)

# Prepare to collect stats
all_speeds = []  # List of (frame_idx, track_id, speed)
frame_idx = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = speedestimator(im0)
    # Log each detected speed
    for track_id, speed in speedestimator.spd.items():
        print(f"Track ID: {track_id}, Speed: {speed} km/h")
        all_speeds.append((frame_idx, track_id, speed))
    video_writer.write(results.plot_im)
    frame_idx += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Write stats to result.md
if all_speeds:
    with open("result.md", "w") as f:
        f.write("| Frame | Track ID | Speed (km/h) |\n")
        f.write("|-------|----------|--------------|\n")
        for frame, track_id, speed in all_speeds:
            f.write(f"| {frame} | {track_id} | {speed} |\n")
        # Calculate stats
        speeds_only = [s for _, _, s in all_speeds]
        fastest = max(speeds_only)
        slowest = min(speeds_only)
        average = sum(speeds_only) / len(speeds_only)
        f.write("\n")
        f.write(f"**Fastest speed:** {fastest} km/h\n\n")
        f.write(f"**Average speed:** {average:.2f} km/h\n\n")
        f.write(f"**Slowest speed:** {slowest} km/h\n")