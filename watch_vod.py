from vod.configuration import KittiLocations
from vod.visualization import Visualization2D

kitti_locations = KittiLocations(root_dir="/DISK_F/view_of_delft_PUBLIC",
                                output_dir="lidar_output/",
                                frame_set_path="",
                                pred_dir="",
                                )

print(f"Lidar directory: {kitti_locations.lidar_dir}")
print(f"Radar directory: {kitti_locations.radar_dir}")

from vod.frame import FrameDataLoader

frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                             frame_number="00393")


# 看图片
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')

# imgplot = plt.imshow(frame_data.image)
# plt.show()


# vis2d = Visualization2D(frame_data)
# vis2d.draw_plot(#show_lidar=True,
#                 plot_figure=False,
#                 show_radar=True,
#                 show_gt=True,
#                 # min_distance_threshold=5,
#                 # max_distance_threshold=20,
#                 save_figure=True)

#看雷达
# print(frame_data.lidar_data)

# 3D Visualization of the point-cloud
from vod.visualization import VisualizationOpen3D
vis_3d = VisualizationOpen3D(frame_data=frame_data,origin="lidar")


vis_3d.draw_plot(lidar_points_plot=True,
                radar_velocity_plot=True,
                radar_points_plot=True,
                write_figure=True,
                annotations_plot=True)