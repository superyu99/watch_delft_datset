from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader


kitti_locations = KittiLocations(root_dir="example_set",
                                output_dir="example_output")

frame_data = FrameDataLoader(kitti_locations=kitti_locations,
                             frame_number="01201")

from vod.visualization import Visualization2D

vis2d = Visualization2D(frame_data)

vis2d.draw_plot()