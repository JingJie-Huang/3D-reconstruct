import open3d as o3d
import numpy as np


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

if __name__ == "__main__":

    print("Testing IO for point cloud ...")
    pcd = o3d.io.read_point_cloud("pointcloud.txt", format = 'xyzrgb')
    print(pcd)
    #print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])
    
    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    o3d.visualization.draw_geometries([voxel_down_pcd])
    
    print("Statistical oulier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30,
                                                        std_ratio=1.0)
    display_inlier_outlier(voxel_down_pcd, ind)
    
    voxel_down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([voxel_down_pcd])
    
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:mesh,densities=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(voxel_down_pcd, depth=9)
    print(mesh)
    o3d.visualization.draw_geometries([mesh], 
                                      zoom=0.664, 
                                      front=[-0.4761, -0.4698, -0.7434], 
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])
    
    
    
    
    
    
   	
