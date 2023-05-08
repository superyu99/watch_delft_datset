import cv2
import os
from tqdm import tqdm

start_indices = []
end_indices = []

with open('/workspace/mot/whatch_delft_dataset/view_of_delft/lidar/ImageSets/full.txt', 'r') as f:
    lines = f.readlines()
    frame_numbers = [int(line.strip()) for line in lines]

start_indices.append(frame_numbers[0])

for i in range(len(frame_numbers) - 1):
    diff = abs(frame_numbers[i + 1] - frame_numbers[i])
    if diff > 1:
        end_indices.append(frame_numbers[i])
        start_indices.append(frame_numbers[i + 1])

end_indices.append(frame_numbers[-1])

assert len(start_indices) == len(end_indices)

image_folder = '/workspace/mot/whatch_delft_dataset/output_image'


def create_video(start_frame, end_frame, video_name, dataset_info):
    # Get the size of the first frame
    first_frame_name = f"{start_frame:05d}.png"
    first_frame_path = os.path.join(image_folder, first_frame_name)
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape
    size = (width, height)

    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

    for frame_number in tqdm(range(start_frame, end_frame + 1), desc=f"Creating {dataset_info} video"):
        img_name = f"{frame_number:05d}.png"
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        # Add frame number and dataset info to the image
        text = f"{frame_number:05d} {dataset_info},start:{start_frame},end:{end_frame}"
        cv2.putText(img, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

        # Write the frame directly to the video file
        out.write(img)

    out.release()

for i in range(0,7):
    create_video(start_indices[i], end_indices[i], f'train_radar_5_{i}.mp4', 'train')
# Combine frames into videos

# create_video(start_indices[7], end_indices[10], 'val.mp4', 'val')
# create_video(start_indices[11], end_indices[14], 'test.mp4', 'test')