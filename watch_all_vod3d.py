from tqdm import tqdm
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from vod.visualization import Visualization2D
from vod.visualization import VisualizationOpen3D

kitti_locations = KittiLocations(root_dir="/workspace/mot/whatch_delft_dataset/view_of_delft",
                                output_dir="/workspace/mot/whatch_delft_dataset/output_image/")



start_indices = []
end_indices = []

with open('/workspace/mot/whatch_delft_dataset/view_of_delft/lidar/ImageSets/full.txt', 'r') as f:
    lines = f.readlines()
    frame_numbers = [int(line.strip()) for line in lines]

for i in range(len(frame_numbers) - 1):
    img_name = f"{frame_numbers[i]:05d}"
    frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                             frame_number=img_name)
    vis_3d = VisualizationOpen3D(frame_data=frame_data,origin="lidar")
    vis_3d.draw_plot(lidar_points_plot=True,
                radar_velocity_plot=True,
                radar_points_plot=True,
                write_figure=True,
                annotations_plot=True)