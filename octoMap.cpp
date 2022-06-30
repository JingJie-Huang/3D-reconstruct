#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Geometry>
#include <octomap/octomap.h>

using namespace std;
using namespace cv;

bool readYaml(Mat &cameraMatrix, Mat &distCoeffs);

int main(int argc, char **argv)
{
    Mat color, color_undist, depth; // rgb and depth images
    Mat cameraMatrix, distCoeffs;
    bool state;

    if (argc != 2)
    {
        cout << "Please ENTER the path to the dataset" << endl;
        cout << "usage: joinMap_modified path_to_dataset" << endl;
        // path_to_dataset: "../rgbd_dataset_freiburg1_360"
        return 1;
    }

    state = readYaml(cameraMatrix, distCoeffs);
    if (!state)
    {
        cout << "Error reading camera .yaml file" << endl;
        return 1;
    }

    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate_all.txt";

    ifstream fin(associate_file);

    string rgb_file, depth_file, time_rgb, time_depth, time_groundtruth;
    string tx, ty, tz, qx, qy, qz, qw;
    double data[7] = {0}; // store quaternion
    int count_num = 0;
    unsigned int index;

    // Intrinsic matrix
    // from TUM Freiburg 1 RGB
    // please refer to: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    // camera parameter: cx = 318.6, cy = 255.3, fx = 517.3, fy = 516.5, depthScale = 5000.0

    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double depthScale = 5000.0;

    // octomap
    octomap::OcTree tree(0.02);

    while (!fin.eof())
    {

        fin >> time_rgb >> rgb_file >> time_depth >> depth_file >> time_groundtruth >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        if (!fin.fail())
        {
            count_num++;

            color = imread(path_to_dataset + "/" + rgb_file);
            undistort(color, color_undist, cameraMatrix, distCoeffs, getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, color.size(), 0, color.size()));
            depth = imread(path_to_dataset + "/" + depth_file, -1); // use -1 to read original image
            // convert string to double (pose)
            data[0] = stod(tx);
            data[1] = stod(ty);
            data[2] = stod(tz);
            data[3] = stod(qx);
            data[4] = stod(qy);
            data[5] = stod(qz);
            data[6] = stod(qw);
            Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
            Eigen::Isometry3d T(q);
            T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
            octomap::Pointcloud cloud; // the point cloud in octomap

            cout << "Converting image NO." << count_num << endl;

            for (int v = 0; v < color_undist.rows; v++)
            {
                for (int u = 0; u < color_undist.cols; u++)
                {

                    if (u % 10 != 0 || v % 10 != 0 || count_num % 10 != 0)
                    {
                        continue; // reduce point cloud size, which increase the 3d drawing process
                    }
                    index = v * color.cols + u;
                    unsigned int d = depth.ptr<unsigned short>(v)[u];

                    if (d == 0)
                        continue; // 0 indicates not measured
                    Eigen::Vector3d point;
                    point[2] = double(d) / depthScale;
                    point[0] = (u - cx) * point[2] / fx;
                    point[1] = (v - cy) * point[2] / fy;
                    Eigen::Vector3d pointWorld = T * point;

                    cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
                }
            }
            tree.insertPointCloud(cloud, octomap::point3d(T(0,3), T(1,3), T(2,3)));
        }
    }

    cout << "There are total number of " << count_num << " data" << endl;
    tree.updateInnerOccupancy();
    cout<<"saving octomap ... "<<endl;
    tree.writeBinary( "octomap.bt" );

    imshow("Original image", color);
    imshow("Undistortion image", color_undist);
    waitKey(0);

    fin.close();

    return 0;
}

bool readYaml(Mat &cameraMatrix, Mat &distCoeffs)
{
    std::string filename = "../camera_calibration_parameters.yaml";
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "failed to open file " << filename << endl;
        return false;
    }

    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    cout << "Reading .yaml file ..." << endl
         << endl;
    cout << "cameraMatrix = \n"
         << cameraMatrix << endl;
    cout << "distCoeffs = \n"
         << distCoeffs << endl;

    // read string
    string timeRead;
    fs["calibrationDate"] >> timeRead;
    cout << "calibrationDate = " << timeRead << endl;
    cout << "Finish reading .yaml file" << endl;

    fs.release();
    return true;
}
