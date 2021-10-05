import open3d as o3d
import os
import numpy as np
import configparser,argparse
import time
#### Script  used to downsample datasets specified in config.ini and cfg/datasets/DATASET.ini


downsamplingScale=4

config = configparser.ConfigParser()
config.read('config.ini')
pathDataset = config["FOLDERS"]["dataset"]

datasetsKeys = list(config["DATASETS"].keys())

def makeDirIfNotExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def downsampleNP(pcdIn,pathOutput,downsamplInt=downsamplingScale):
    pcdNew = o3d.geometry.PointCloud()
    pcdNew.points = o3d.utility.Vector3dVector(np.asarray(pcdIn.points)[::downsamplInt])
    pcdNew.colors = o3d.utility.Vector3dVector(np.asarray(pcdIn.colors)[::downsamplInt])
    ##### You can vizualize the results here
    #o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.draw_geometries([pcdNew])
    o3d.io.write_point_cloud(pathOutput, pcdNew,write_ascii=True)
    with open(pathOutput) as temp:
        lines = temp.readlines()

    lines[4] = lines[4].replace('double', 'float')
    lines[5] = lines[5].replace('double', 'float')
    lines[6] = lines[6].replace('double', 'float')

    writer = open(pathOutput, "w+")

    for line in lines:
        writer.write(line)


for datasetKey in datasetsKeys:
    datasetConfig = configparser.ConfigParser()
    datasetConfig.read(os.path.join("cfg/datasets", datasetKey + ".ini"))
    pointCloudKeys = list(datasetConfig["POINTCLOUDS"].keys())
    for pointCloudKey in pointCloudKeys:
        print("Downsampling dataset: ", datasetKey," PointCloud: ", pointCloudKey)
        pathPointCloudSequence = os.path.join(pathDataset,datasetConfig['BASE']['baseDir'],datasetConfig['POINTCLOUDS'][pointCloudKey])

        pathPointCloudSequenceOutput = os.path.join(pathDataset, datasetConfig['BASE']['baseDir']+"_downsampled_scale_"+str(downsamplingScale),
                                              datasetConfig['POINTCLOUDS'][pointCloudKey])
        makeDirIfNotExist(pathPointCloudSequenceOutput)
        _, _, files = next(os.walk(pathPointCloudSequence))
        timeStart = time.time()
        for file in files:
            print("Downsampling file: ",file," Process Time: ", time.time()-timeStart)
            timeStart = time.time()
            pcd = o3d.io.read_point_cloud(os.path.join(pathPointCloudSequence, file))
            downsampleNP(pcd,pathOutput=os.path.join(pathPointCloudSequenceOutput,file))
