import open3d as o3d
import os

# 创建一个简单的坐标轴
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)

# 创建一个离屏渲染器
width, height = 640, 480
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

# 创建一个场景
scene = o3d.visualization.rendering.Open3DScene(renderer)

# 添加几何体到场景
scene.add_geometry("mesh", mesh, o3d.visualization.rendering.Material())

# 设置相机参数
camera_params_file = "sttracker-mrt-view.json"
if camera_params_file is not None and os.path.exists(camera_params_file):
    # 从JSON文件中读取相机参数
    param = o3d.io.read_pinhole_camera_parameters(camera_params_file)
    # 创建一个相机并应用参数
    camera = o3d.visualization.rendering.Camera()
    camera.set_projection(param.intrinsic.intrinsic_matrix, 0.5, 1000)
    camera.look_at(param.extrinsic[:3, 3], param.extrinsic[:3, 3] + param.extrinsic[:3, 2], -param.extrinsic[:3, 1])
    scene.set_active_camera(camera)

# 渲染并保存图片
image = o3d.visualization.rendering.Image(width, height)
renderer.render_to_image(scene, image)
o3d.io.write_image("output_image.png", image)