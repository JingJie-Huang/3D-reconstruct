import open3d as o3d
import numpy as np

if __name__ == "__main__":

    print("Testing IO for point cloud ...")
    pcd = o3d.io.read_point_cloud("pointcloud.txt", format = 'xyzrgb')
    print(pcd)
    #print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])
    
   	
