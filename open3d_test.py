import open3d as o3d

vis = o3d.visualization.Visualizer()
vis.create_window()
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
vis.add_geometry(mesh)
vis.run()
vis.destroy_window()