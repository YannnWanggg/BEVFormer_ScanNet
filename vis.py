import torch
import open3d
import json
import cv2
import pyrender
import trimesh

import numpy as np


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis, boxes3d = draw_box(vis, gt_boxes, (0, 1, 0))

    if ref_boxes is not None:
        vis, boxes3d = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

    return boxes3d


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    if isinstance(gt_boxes, dict):
        center = gt_boxes["center"]
        lwh = gt_boxes["dimensions"]
        rot = gt_boxes["rotation_matrix"]
    else:
        center = gt_boxes[0:3]
        lwh = gt_boxes[3:6]
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    boxes3d = []
    for i in range(len(gt_boxes)):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # corners = np.array(box3d.get_box_points())
        # box_corners.append(corners)
        boxes3d.append(box3d)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis, boxes3d


def get_box(objects_info):
    scene_id = "scene0000_00"
    objs_info = objects_info[scene_id]["objects_info"]

    bboxes = []
    for obj_info in objs_info:
        position = obj_info['position']
        size = obj_info['size']
        bbox = position + size + [0]
        bboxes.append(bbox)
    
    bboxes = np.array(bboxes)

    return bboxes


def render_ply_with_pose(mesh_file: pyrender.Mesh, camera_pose, intrinsics, image_size):
    """
    Render a mesh with a given camera pose using pyrender.

    Args:
        mesh_file: trimesh file
        camera_pose (np.ndarray): 4x4 array for camera pose (world to camera transformation).
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        image_size (tuple): Size of the output image (height, width).

    Returns:
        np.ndarray: Rendered image as a numpy array (H x W x 3).
    """
    # if not mesh_file.is_empty:
    #     vertices = mesh_file.vertices
    #     colors = mesh_file.visual.vertex_colors[:, :3] / 255.0
    
    # cloud_mesh = pyrender.Mesh.from_points(
    #     vertices, colors=colors
    # )
    mesh = pyrender.Mesh.from_trimesh(mesh_file)

    # Create a Pyrender Scene
    scene = pyrender.Scene()
    # scene.add(cloud_mesh)
    scene.add(mesh)

    # Add the camera to the scene
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    scene.add(camera, pose=camera_pose)

    # Create a light source
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # Render the scene
    r = pyrender.OffscreenRenderer(*image_size)
    color, _ = r.render(scene)

    return color


def render_point_cloud_with_pose(point_cloud, colors, camera_pose, intrinsics, image_size):
    """
    Render a point cloud with a given camera pose using pyrender.

    Args:
        point_cloud (np.ndarray): Nx6 array of 3D points and colors (x, y, z, r, g, b).
        camera_pose (np.ndarray): 4x4 array for camera pose (world to camera transformation).
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        image_size (tuple): Size of the output image (height, width).

    Returns:
        np.ndarray: Rendered image as a numpy array (H x W x 3).
    """
    # Normalize colors to [0, 1] if necessary
    # colors = point_cloud[:, 3:] / 255.0 if point_cloud[:, 3:].max() > 1.0 else point_cloud[:, 3:]

    # Create a Trimesh Point Cloud
    trimesh_points = trimesh.points.PointCloud(point_cloud[:, :3], colors=colors)

    # Create a Pyrender Mesh for the point cloud
    cloud_mesh = pyrender.Mesh.from_points(
        point_cloud[:, :3], colors=colors
    )

    # Create a Pyrender Scene
    scene = pyrender.Scene()
    scene.add(cloud_mesh)

    # Add the camera to the scene
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    scene.add(camera, pose=camera_pose)

    # Create a light source
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # Render the scene
    r = pyrender.OffscreenRenderer(*image_size)
    color, _ = r.render(scene)

    return color
    

objects_info_path = "/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scannet_ssg/scene0000_00/objects.json"
pcd_path = "/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scan_data/pcd_with_global_alignment/scene0000_00.pth"

pcd_data = torch.load(pcd_path)
points, ori_colors, _, _ = pcd_data
colors = ori_colors / 255

with open(objects_info_path) as f:
    objects_info = json.load(f)

gt_bboxes = get_box(objects_info)


axis_align_matrices_path = "/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/annotations/meta_data/scans_axis_alignment_matrices.json"
with open(axis_align_matrices_path) as f:
    axis_align_matrices = json.load(f)

axis_align_matrix = np.array(axis_align_matrices["scene0000_00"]).reshape(4, 4)
c2w_pose = np.loadtxt("/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/ScanNetV2-RGBD/org_frame_data/scene0000_00/scene0000_00_pose/1322.txt")
c2w_pose = np.matmul(axis_align_matrix, c2w_pose)
# c2w_pose = np.matmul(pose, axis_align_matrix)
# w2c_pose = np.linalg.inv(c2w_pose)

# rotation = np.array([
#     [-1., 0., 0., 0.], 
#     [0., 1., 0., 0.], 
#     [0., 0., -1., 0.], 
#     [0., 0., 0., 1.]
# ])
# pose = np.matmul(c2w_pose, rotation)
# rotation2 = np.array([
#     [-1., 0., 0., 0.], 
#     [0., -1., 0., 0.], 
#     [0., 0., 1., 0.], 
#     [0., 0., 0., 1.]
# ])
# pose = np.matmul(pose, rotation2)

intrinsic_path = "/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/ScanNetV2-RGBD/org_frame_data/scene0000_00/scene0000_00_intrinsic/intrinsic_depth.txt"
ori_intrinsic = np.array(np.loadtxt(intrinsic_path))
intrinsic = ori_intrinsic[:3, :3]

image_size = (640, 480)

# rendered_image = render_point_cloud_with_pose(points, colors, pose, intrinsic, image_size)


mesh_path = "/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.rot.ply"
mesh_file = trimesh.load(mesh_path)

# rendered_image = render_ply_with_pose(mesh_file, pose, intrinsic, image_size)

# # Visualize the rendered image
# import matplotlib.pyplot as plt
# plt.imshow(rendered_image)
# plt.axis("off")
# plt.show()

draw_scenes(points, gt_boxes=gt_bboxes, point_colors=colors)


def transform_point_cloud_to_camera(point_cloud, camera_pose):
    """
    将点云从世界坐标系变换到相机坐标系。
    
    参数:
    - point_cloud: 形状为 (N, 3) 的点云数据，其中 N 是点的数量
    - camera_pose: 4x4 相机外参矩阵，表示从相机到世界的变换
    
    返回:
    - transformed_points: 变换后的点云数据，形状为 (N, 3)
    """
    # 确保点云是 Nx3 形状
    assert point_cloud.shape[1] == 3
    
    # 将点云转换为齐次坐标 (N, 4)，在最后一列添加 1
    points_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    
    # 计算世界到相机的变换矩阵，即相机外参的逆矩阵
    camera_to_world_inv = np.linalg.inv(camera_pose)
    
    # 将点云从世界坐标系转换到相机坐标系
    transformed_points_homogeneous = points_homogeneous @ camera_to_world_inv.T
    
    # 只取前 3 个坐标值，返回变换后的点云
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points


transformed_points = transform_point_cloud_to_camera(points, c2w_pose)

def transform_bounding_boxes_3d_z_rotation(boxes, camera_pose):
    """
    对 N 个 3D bounding boxes 进行变换，仅考虑绕 z 轴的旋转，包括中心点和旋转角度。

    参数:
    - boxes: ndarray, 大小为 (N, 7)，每个 bounding box 格式为 (center_x, center_y, center_z, w, h, l, 0)
    - int_mat: ndarray, 相机的内参矩阵
    - camera_pose: ndarray, 相机的外参矩阵, camera2world

    返回:
    - transformed_boxes: list of dicts, 包含变换后的所有 bounding box
    """
    transformed_boxes = []
    
    # 提取外参的旋转矩阵（这里只需要考虑 z 轴旋转部分）
    camera_to_world_inv = np.linalg.inv(camera_pose)
    rotation_matrix_world_to_cam = camera_to_world_inv[:3, :3]  # 外参的旋转部分

    # 只提取 z 轴旋转角度
    angle_z = np.arctan2(rotation_matrix_world_to_cam[1, 0], rotation_matrix_world_to_cam[0, 0])

    # 逐个处理每个 bounding box
    for box in boxes:
        # 提取 bounding box 中心点
        center = np.array([box[0], box[1], box[2], 1]).reshape(4, 1)

        # # 计算投影矩阵
        # intrinsic_4x4 = np.identity(4)
        # intrinsic_4x4[:3, :3] = int_mat
        # proj_mat = np.matmul(intrinsic_4x4, np.linalg.inv(pose))

        # 变换中心点
        # transformed_center = np.dot(proj_mat, center).flatten()
        transformed_center = np.dot(camera_to_world_inv, center).flatten()
        transformed_center /= transformed_center[3]  # 归一化为非齐次坐标

        # initial bounding orientation
        initial_rotation_matrix = np.eye(3)

        # calculate the rotation matrix after transformation
        transformed_rotation_matrix = np.dot(rotation_matrix_world_to_cam, initial_rotation_matrix)

        # 构造变换后的 bounding box
        transformed_box = [
            transformed_center[0],  # transformed center x
            transformed_center[1],  # transformed center y
            transformed_center[2],  # transformed center z
            box[3],  # width w
            box[4],  # height h
            box[5],  # length l
            angle_z  # rotation_z (仅考虑绕 z 轴旋转)
        ]

        transformed_boxes.append(transformed_box)

    return np.array(transformed_boxes)


def transform_bounding_boxes_3d(boxes, camera_pose):
    """
    对 N 个 3D bounding boxes 进行变换，包括中心点和旋转角度。

    参数:
    - boxes: ndarray, 大小为 (N, 7)，每个 bounding box 格式为 (center_x, center_y, center_z, w, h, l, 0)
    - int_mat: ndarray, 相机的内参矩阵
    - camera_pose: ndarray, 相机的外参矩阵, camera2world

    返回:
    - transformed_boxes: list of dicts, 包含变换后的所有 bounding box
    """
    transformed_boxes = []
    
    # 提取外参的旋转矩阵（这里只需要考虑 z 轴旋转部分）
    camera_to_world_inv = np.linalg.inv(camera_pose)
    rotation_matrix_world_to_cam = camera_to_world_inv[:3, :3]  # 外参的旋转部分

    # 只提取 z 轴旋转角度
    angle_z = np.arctan2(rotation_matrix_world_to_cam[1, 0], rotation_matrix_world_to_cam[0, 0])

    # 逐个处理每个 bounding box
    for box in boxes:
        # 提取 bounding box 中心点
        center = np.array([box[0], box[1], box[2], 1]).reshape(4, 1)

        # # 计算投影矩阵
        # intrinsic_4x4 = np.identity(4)
        # intrinsic_4x4[:3, :3] = int_mat
        # proj_mat = np.matmul(intrinsic_4x4, np.linalg.inv(pose))

        # 变换中心点
        # transformed_center = np.dot(proj_mat, center).flatten()
        transformed_center = np.dot(camera_to_world_inv, center).flatten()
        transformed_center /= transformed_center[3]  # 归一化为非齐次坐标

        # initial bounding orientation
        initial_rotation_matrix = np.eye(3)

        # calculate the rotation matrix after transformation
        transformed_rotation_matrix = np.dot(rotation_matrix_world_to_cam, initial_rotation_matrix)

        # 构造变换后的 bounding box
        transformed_box = {
            "center": transformed_center[:3], # x, y, z
            "dimensions": (box[3], box[4], box[5]), 
            "rotation_matrix": transformed_rotation_matrix
        }

        transformed_boxes.append(transformed_box)

    return transformed_boxes



# transformed_boxes = transform_bounding_boxes_3d_z_rotation(gt_bboxes, c2w_pose)
transformed_boxes = transform_bounding_boxes_3d(gt_bboxes, c2w_pose)

boxes3d = draw_scenes(transformed_points, gt_boxes=transformed_boxes, point_colors=colors)


def project_points_to_image(pts_3d, intrinsic_matrix):
    """
    将 3D 点投影到图像平面
    """
    # 将点转换到齐次坐标
    pts_3d_homogeneous = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    # 投影到图像平面
    pts_img_homogeneous = np.dot(intrinsic_matrix, pts_3d_homogeneous.T).T
    pts_img = pts_img_homogeneous[:, :2] / pts_img_homogeneous[:, 2].reshape(-1, 1)
    
    return pts_img


def is_box_in_image(boxes3d, intrinsic_matrix, image_width, image_height):
    results = []
    for box in boxes3d:
        corners_3d = np.asarray(box.get_box_points())
        
        projected_corners = project_points_to_image(corners_3d, intrinsic_matrix)

        flag = False
        for point in projected_corners:
            x, y = point
            if x < 0 or x >= image_width or y < 0 or y >= image_height:
                continue
            else:
                flag = True
    
        results.append(flag)

    return results


boxes_in_image = is_box_in_image(boxes3d, ori_intrinsic, 640, 480)

current_boxes = []
for i, box in enumerate(transformed_boxes):
    if boxes_in_image[i] and box["center"][2] > 0:
        current_boxes.append(box)

boxes3d = draw_scenes(transformed_points, gt_boxes=current_boxes, point_colors=colors)