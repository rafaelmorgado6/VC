import open3d as o3d
import numpy as np
import copy

# Importando a subbiblioteca de registro de ICP
from open3d.pipelines import registration

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

# Função para seleção de pontos (caso seja necessário para inicializar o ICP)
def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press q to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # usuário escolhe os pontos
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

# Carregar as nuvens de pontos
pcd_office1 = o3d.io.read_point_cloud('..//depth_images//office1.pcd')
pcd_office2 = o3d.io.read_point_cloud('..//depth_images//office2.pcd')

# Remoção de pontos não finitos (valores NaN)
pcd_office1.remove_non_finite_points()
pcd_office2.remove_non_finite_points()

# Amostragem das nuvens de pontos usando o filtro voxel_down_sample
voxel_size = 0.01
pcd_office1 = pcd_office1.voxel_down_sample(voxel_size=voxel_size)
pcd_office2 = pcd_office2.voxel_down_sample(voxel_size=voxel_size)

# Definindo parâmetros ICP e realizando o registro
print("Iniciando ICP para alinhamento das nuvens de pontos...")

# Executando o ICP para alinhar as nuvens de pontos
threshold = 0.02  # Distância máxima entre os pontos para considerar a correspondência
icp_result = registration.registration_icp(
    pcd_office1, pcd_office2, threshold,
    np.eye(4),  # Transformação inicial (identidade)
    registration.TransformationEstimationPointToPoint())

# Obtendo a transformação calculada
print("Transformação calculada pelo ICP:")
print(icp_result.transformation)

# Aplicando a transformação para alinhar a primeira nuvem de pontos
pcd_office1.transform(icp_result.transformation)

# Visualizando a nuvem de pontos original e a alinhada
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

draw_registration_result(pcd_office1, pcd_office2, icp_result.transformation)

# Mesclando as duas nuvens de pontos alinhadas
merged_pcd = pcd_office1 + pcd_office2

# Salvando a nuvem de pontos mesclada em um arquivo PLY
o3d.io.write_point_cloud("merged_offices.ply", merged_pcd)

print("Nuvem de pontos mesclada salva como 'merged_offices.ply'.")