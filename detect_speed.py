import cv2
import numpy as np

from ultralytics import solutions

cap = cv2.VideoCapture("video/federal.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("output_speed_estimated_federal.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize SpeedEstimator
speedestimator = solutions.SpeedEstimator(
    model="yolo12s.pt",
    show=True,
)

# Define rectangles as (x1, y1, x2, y2)
RECT1 = (400, 430, 1500, 540)
RECT2 = (400, 550, 1500, 650)
ROT_ANGLE = -4.5  # degrees

# Convert (x1, y1, x2, y2) to rotated rectangle definition ((cx, cy), (w, h), angle)
def rect_to_rotated_def(rect, angle):
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return ((cx, cy), (w, h), angle)

RECT1_DEF = rect_to_rotated_def(RECT1, ROT_ANGLE)
RECT2_DEF = rect_to_rotated_def(RECT2, ROT_ANGLE)

def get_rotated_rect_points(rect_def):
    rect = (rect_def[0], rect_def[1], rect_def[2])
    box = cv2.boxPoints(rect)
    return np.int32(box)

def point_in_rotated_rect(point, rect_def):
    rect = (rect_def[0], rect_def[1], rect_def[2])
    box = cv2.boxPoints(rect)
    return cv2.pointPolygonTest(box, point, False) >= 0

# Prepare to collect stats for each region
all_speeds = []  # (frame_idx, track_id, speed, region_label)
frame_idx = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = speedestimator(im0)
    # Draw rotated rectangles for visualization
    rect1_pts = get_rotated_rect_points(RECT1_DEF)
    rect2_pts = get_rotated_rect_points(RECT2_DEF)
    cv2.polylines(im0, [rect1_pts], isClosed=True, color=(0,255,0), thickness=2)
    cv2.polylines(im0, [rect2_pts], isClosed=True, color=(255,0,0), thickness=2)
    # For each detected speed, check which region the object is in
    if hasattr(speedestimator, 'boxes') and hasattr(speedestimator, 'track_ids'):
        for box, track_id in zip(speedestimator.boxes, speedestimator.track_ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            speed = speedestimator.spd.get(track_id)
            region_label = None
            if speed is not None:
                if point_in_rotated_rect((cx, cy), RECT1_DEF):
                    region_label = 'Heading To kl'
                elif point_in_rotated_rect((cx, cy), RECT2_DEF):
                    region_label = 'Heading To klang'
                if region_label:
                    all_speeds.append((frame_idx, track_id, speed, region_label))
                    print(f"Frame {frame_idx} | Track ID: {track_id} | Speed: {speed} km/h | Region: {region_label}")
    # Show the frame with rectangles
    cv2.imshow('Speed Detection with Regions', im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    video_writer.write(results.plot_im)
    frame_idx += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()

def write_stats_simple(filename, region_name, all_speeds):
    region_speeds = [row for row in all_speeds if row[3] == region_name]
    with open(filename, "w") as f:
        if region_speeds:
            speeds_only = [s for _, _, s, _ in region_speeds]
            fastest = max(speeds_only)
            slowest = min(speeds_only)
            average = sum(speeds_only) / len(speeds_only)
            f.write(f"# {region_name}\n\n")
            f.write(f"**Fastest speed:** {fastest} km/h\n\n")
            f.write(f"**Average speed:** {average:.2f} km/h\n\n")
            f.write(f"**Slowest speed:** {slowest} km/h\n\n")
            f.write("## Individual Speeds\n\n")
            f.write("| Frame | Track ID | Speed (km/h) |\n")
            f.write("|-------|----------|--------------|\n")
            for frame, track_id, speed, _ in region_speeds:
                f.write(f"| {frame} | {track_id} | {speed} |\n")
        else:
            f.write(f"# {region_name}\nNo objects detected in this region.\n\n")

write_stats_simple("result_to_kl.md", "Heading To kl", all_speeds)
write_stats_simple("result_to_klang.md", "Heading To klang", all_speeds)

# Queue ID: ae9fae9b-ed60-4adc-9c7b-399cb41c5b58

# https://bookmyshowsg.queue-it.net/?c=bookmyshowsg&e=splanet20250529gd&ver=javascript-4.3.1&cver=2156&man=splanet-booking-GD2025KL&t=https%3A%2F%2Fstarplanet.bigtix.io%2Fbooking%2FGD2025KL%3Fotm%3Dx7f2b9qk&kupver=cloudflare-4.2.2
# https://bookmyshowsg.queue-it.net/?c=bookmyshowsg&e=splanet20250529gd&t=https%3A%2F%2Fstarplanet.bigtix.io%2Fbooking%2FGD2025KL%3Fotm%3Dx7f2b9qk&cid=en-US


# https://bookmyshowsg.queue-it.net/?c=bookmyshowsg&e=splanet20250529gd&q=3835ee17-e3ae-4d2e-adfc-