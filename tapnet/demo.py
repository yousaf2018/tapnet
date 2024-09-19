import cv2
import numpy as np
import jax
import matplotlib
import matplotlib.pyplot as plt
from tapnet.models import tapir_model
from tapnet.utils import transforms, viz_utils, model_utils

matplotlib.use('Agg')

# Load Model Checkpoint
MODEL_TYPE = 'bootstapir'  # 'tapir' or 'bootstapir'

print("Loading checkpoint...")

if MODEL_TYPE == 'tapir':
    checkpoint_path = 'C:\\Users\\neurog\\Documents\\tapnet\\checkpoints\\causal_tapir_checkpoint.npy'
else:
    checkpoint_path = 'C:\\Users\\neurog\\Documents\\tapnet\\checkpoints\\causal_bootstapir_checkpoint.npy'
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state['params'], ckpt_state['state']

kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
if MODEL_TYPE == 'bootstapir':
    kwargs.update(dict(
        pyramid_level=1,
        extra_convs=True,
        softmax_temperature=10.0
    ))

tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)

print("Checkpoint loaded successfully.")

# Load video using OpenCV
video_path = 'C:\\Users\\neurog\\Documents\\tapnet\\assets\\test_video.mp4'
cap = cv2.VideoCapture(video_path)

frames = []
print("Reading video frames...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

video = np.array(frames)  # Convert list to numpy array
cap.release()
print(f"Video loaded with {len(frames)} frames.")

height, width = video.shape[1:3]
print(f"Original video dimensions: {height}x{width}")

# List to store clicked points
clicked_points = []

# Mouse callback function to store click coordinates
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point selected at (x={x}, y={y})")
        clicked_points.append([param, y, x])  # Add a point with (time, y, x)

# Set up the OpenCV window and bind the mouse callback
cv2.namedWindow('Video Frame')
cv2.setMouseCallback('Video Frame', select_point)

# Utility Functions
def inference(frames, query_points):
    """Inference on one video.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8
      query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

    Returns:
      tracks: [num_points, 3], [-1, 1], [t, y, x]
      visibles: [num_points, num_frames], bool
    """
    print("Starting inference...")

    # Preprocess video to match model inputs format
    frames = model_utils.preprocess_frames(frames)
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    outputs = tapir(video=frames, query_points=query_points, is_training=False, query_chunk_size=32)
    tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

    # Binarize occlusions
    visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)

    print("Inference complete.")
    return tracks[0], visibles[0]

inference = jax.jit(inference)

# Predict Sparse Point Tracks
resize_height = 256  # Resize height
resize_width = 256   # Resize width

print(f"Resizing video to {resize_width}x{resize_height}...")
frames_resized = np.array([cv2.resize(frame, (resize_width, resize_height)) for frame in video])

# Indices for point selection
first_frame_idx = 0
middle_frame_idx = len(frames_resized) // 2
last_frame_idx = len(frames_resized) - 1

# Show first frame and wait for user to select points
print("Select points by clicking on the first frame. Press 'q' when done.")
cv2.setMouseCallback('Video Frame', select_point, param=first_frame_idx)  # Frame 0 for initial points
while True:
    cv2.imshow('Video Frame', frames_resized[first_frame_idx])
    if cv2.waitKey(1) & 0xFF == ord('q') or len(clicked_points) >= 200:
        break

# Show middle frame and allow for point selection
print(f"Select points by clicking on the middle frame ({middle_frame_idx}). Press 'q' when done.")
cv2.setMouseCallback('Video Frame', select_point, param=middle_frame_idx)  # Middle frame for points
while True:
    cv2.imshow('Video Frame', frames_resized[middle_frame_idx])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Show last frame and allow for point selection
print(f"Select points by clicking on the last frame ({last_frame_idx}). Press 'q' when done.")
cv2.setMouseCallback('Video Frame', select_point, param=last_frame_idx)  # Last frame for points
while True:
    cv2.imshow('Video Frame', frames_resized[last_frame_idx])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
clicked_points = np.array(clicked_points)
print(f"{len(clicked_points)} points selected.")

# Run inference on the selected points
print("Running inference on the resized frames...")
tracks, visibles = inference(frames_resized, clicked_points)
tracks = np.array(tracks)
visibles = np.array(visibles)

print("Transforming and visualizing tracks...")
# Visualize sparse point tracks
tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
video_viz = viz_utils.plot_tracks_v2(video, tracks, np.logical_not(visibles))

# Save the processed video
output_path = 'C:\\Users\\neurog\\Documents\\tapnet\\result.mp4'
print(f"Saving the video to {output_path}...")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

for frame in video_viz:
    out.write(frame)

out.release()
print("Video saved successfully.")
