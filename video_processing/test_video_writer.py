import cv2

input_path = "data/fightclub.mp4"
output_path = "data/output_video.avi"  # AVI container with XVID codec

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Cannot open input video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    print("Error: Cannot open output video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: add some overlay text to test
    cv2.putText(frame, "Test Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print(f"Video saved at {output_path}")