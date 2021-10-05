import open3d as o3d
import numpy as np
import os
import configparser
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KDTree

def makeDirIfNotExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

_,_,datasetNames = next(os.walk("cfg/datasets"))
datasetNames = [os.path.splitext(x)[0] for x in datasetNames]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset to visualize')
parser.add_argument('--recompute', action="store_true", help="Recompute the values for the dataset")
parser.add_argument('--skipComputation', action="store_true", help="Skip all metric computations and only do visualization")
args = parser.parse_args()

if not args.skipComputation:
    if not args.dataset:
        print("ERROR: Provide the name of a dataset to visualize")
        exit()

    if not args.dataset in datasetNames:
        print("\nERROR: Not a valid dataset \nValid datasets are: \n \n")
        for datasetName in datasetNames:
            print(datasetName)
        exit()

mainConfig = configparser.ConfigParser()
mainConfig.read("config.ini")

if not args.skipComputation:
    configPath = os.path.join("cfg/datasets/", args.dataset + ".ini")
    datasetConfig = configparser.ConfigParser()
    datasetConfig.read(configPath)
    outputFolder = os.path.join("datasetVisualization", args.dataset)
    for key in datasetConfig['POINTCLOUDS']:

        outputFile = os.path.join(outputFolder, str(key) + ".csv")
        computeFile = False
        if not os.path.isfile(outputFile):
            computeFile = True
            print("No result file exist for ", key)
        if args.recompute:
            computeFile = True

        if computeFile:
            makeDirIfNotExist(outputFolder)

            inputPath = os.path.join(mainConfig['FOLDERS']['dataset'], datasetConfig['BASE']['baseDir'], datasetConfig['POINTCLOUDS'][key])
            _,_,files = next(os.walk(inputPath))
            files.sort()

            print("Found ", len(files), " frames for point cloud sequence ", key, "\nAnalyzing...\n")

            noPoints = []
            distances = []
            for i in tqdm(range(len(files))):
                pcdPath = os.path.join(inputPath, files[i])
                pcd = o3d.io.read_point_cloud(pcdPath)

                points = np.asarray(pcd.points)
                noPoints.append(points.shape[0])

                kdTree = KDTree(points, leaf_size=30, metric='euclidean')

                distanceArr, _ = kdTree.query(points, k=2)
                distances.append(np.mean(distanceArr[:,1]))

                #print(i, ": ", np.mean(np.array(noPoints)), np.mean(np.array(distances)))

            noPoints = np.array(noPoints)
            distances = np.array(distances)
            results = np.transpose(np.vstack((noPoints,distances)))
            print("Saving results to: ", outputFile)
            np.savetxt(outputFile, results, delimiter=",")
            print()



metrics = []
dist, points, names = [], [], []

basePath = "datasetVisualization"
_, dirs, _ = next(os.walk(basePath))
dirs.sort()
for dir in dirs:
    currentFolder = os.path.join(basePath, dir)
    _, _, files = next(os.walk(currentFolder))

    datasetPoints = []
    datasetDist = []
    for file in files:
        file = os.path.join(currentFolder, file)
        values = np.genfromtxt(file,delimiter=',')
        values = values.reshape((-1,2))
        pointsMean = np.mean(values[:,0])
        distMean = np.mean(values[:,1])
        datasetPoints.append(pointsMean)
        datasetDist.append(distMean)

    datasetPoints = np.array(datasetPoints)
    datasetDist = np.array(datasetDist)
    dist.append(datasetDist)
    points.append(datasetPoints)
    names.append(dir)

    #metrics.append({'Points': np.mean(datasetPoints), 'PointsStd': np.std(datasetPoints), 'Density': np.mean(datasetDist), 'DensityStd': np.std(datasetDist)})

print(metrics)
#fig = plt.figure(figsize =(10, 7))
fig = plt.figure()


# Creating axes instance
#ax = fig.add_axes([0, 0, 1, 1])

# Creating plot
bp = plt.boxplot(dist, labels=names)
plt.title("Surface density box plot")
plt.xlabel("Dataset")
plt.xticks(rotation=45)
plt.ylabel("Surface density (mean distance between occupied voxels)")
#plt.grid('minor')
fig.tight_layout()

# show plot
plt.savefig(os.path.join(basePath, "surfaceDensity.pdf"))
plt.show()
