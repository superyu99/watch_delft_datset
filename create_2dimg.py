from tqdm import tqdm
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from vod.visualization import Visualization2D

kitti_locations = KittiLocations(root_dir="/workspace/mot/whatch_delft_dataset/view_of_delft",
                                output_dir="/workspace/mot/whatch_delft_dataset/output_image/")



start_indices = []
end_indices = []

with open('/workspace/mot/whatch_delft_dataset/view_of_delft/lidar/ImageSets/full.txt', 'r') as f:
    lines = f.readlines()
    frame_numbers = [int(line.strip()) for line in lines]

for i in tqdm(range(len(frame_numbers) - 1)):
    img_name = f"{frame_numbers[i]:05d}"
    
    frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                             frame_number=img_name)
    vis2d = Visualization2D(frame_data)
    vis2d.draw_plot(#show_lidar=True,
                    plot_figure=False,
                    show_radar=True,
                    show_gt=True,
                    # min_distance_threshold=5,
                    # max_distance_threshold=20,
                    save_figure=True)