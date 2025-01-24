import numpy as np
import open3d as o3d


calibration_params = np.load("3d_coordinates.npz")
points_3D = calibration_params['points']

p = points_3D.reshape(-1, 3)
fp = []

for i in range(p.shape[0]):
    if np.all(~np.isinf(p[i])) and np.all(~np.isnan(p[i])):
        fp.append(p[i])

# Converter lista filtrada para array numpy
fp = np.array(fp)

pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(fp)

# Cropping the mesh using its bouding box to remove positive Z-axis between 0.1 and 5
bbox = pcl.get_axis_aligned_bounding_box()
bbox_points = np.asarray(bbox.get_box_points())
bbox_points[:, 2] = np.clip(bbox_points[:, 2], a_min=0.1, a_max=5)
bbox_cropped = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox_points))
mesh_cropped = pcl.crop(bbox_cropped)

# Create axes mesh
Axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1)

# shome meshes in view
o3d.visualization.draw_geometries([pcl , Axes])