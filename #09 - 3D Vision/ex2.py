import numpy as np
import open3d as o3d

# Leitura das nuvens de pontos dos arquivos PCD
pcd_office1 = o3d.io.read_point_cloud('..//depth_images//office1.pcd')
pcd_office2 = o3d.io.read_point_cloud('..//depth_images//office2.pcd')

pcd_filt_office1 = o3d.io.read_point_cloud('..//depth_images//filt_office1.pcd')
pcd_filt_office2 = o3d.io.read_point_cloud('..//depth_images//filt_office2.pcd')

# Remoção de pontos não finitos (valores NaN)
pcd_office1.remove_non_finite_points()
pcd_office2.remove_non_finite_points()

# Amostragem das nuvens de pontos usando o filtro voxel_down_sample
voxel_size = 0.01
pcd_office1 = pcd_office1.voxel_down_sample(voxel_size=voxel_size)
pcd_office2 = pcd_office2.voxel_down_sample(voxel_size=voxel_size)

# Criar eixos para visualização
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

# Visualização das nuvens de pontos
print("Visualizando office1.pcd")
o3d.visualization.draw_geometries([pcd_office1, axes])

print("Visualizando office2.pcd")
o3d.visualization.draw_geometries([pcd_office2, axes])

# Visualização das nuvens de pontos
print("Visualizando filt_office1.pcd")
o3d.visualization.draw_geometries([pcd_filt_office1, axes])

print("Visualizando filt_office2.pcd")
o3d.visualization.draw_geometries([pcd_filt_office2, axes])