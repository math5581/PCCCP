import open3d as o3d
import metricUtils
import numpy as np
import time
import configparser
import os
import argparse

_,_,codecNames = next(os.walk("cfg/codecs"))
codecNames = [os.path.splitext(x)[0] for x in codecNames]


#arguments. Used to know which of the methods to perform analysis on....
parser = argparse.ArgumentParser()
parser.add_argument('--codec', type=str, help='Codec to compress with')
parser.add_argument('--quality', type=str, help='Compression quality [lowest, low, medium, best, all (execute all configs)]')
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()
args.codec = args.codec.lower()
args.quality = [args.quality]

if args.codec not in codecNames:
    print("Not a valid codec")
    print("Valid codecs: ")
    [print("\t: ", x) for x in codecNames]
    exit()

# Gathering a list of all valid quality arguments
_,_, qualityConfigs = next(os.walk(os.path.join("cfg/codecs", args.codec)))
qualityConfigs.sort()
qualityConfigs.append("all")

#reading quality argument.
if args.quality[0] not in qualityConfigs:
    print(args.quality[0], " is an invalid --quality argument")
    print("Valid quality parameters: ")
    [print("\t: ", x) for x in qualityConfigs]
    exit()

if args.quality[0] == 'all':
    qualityConfigs.pop()
    args.quality = qualityConfigs

#reading codec argument.
configPath = ""
if args.codec == "draco" or args.codec == "Draco":
    args.codec = "draco"
    configPath = 'cfg/codecs/draco.ini'
    c = configparser.ConfigParser()
    c.read(configPath)

elif args.codec == "gpcc" or args.codec == "GPCC":
    args.codec = "gpcc"
    configPath = 'cfg/codecs/gpcc.ini'
    c = configparser.ConfigParser()
    c.read(configPath)

elif args.codec == "vpcc" or args.codec == "VPCC":
    args.codec = "vpcc"
    configPath = 'cfg/codecs/vpcc.ini'
    c = configparser.ConfigParser()
    c.read(configPath)

#Specify here. Only for internal use since I cannot get geometry files for VPCC
if args.codec == "vpcc":
    geometry=True
else:
    geometry=True

#Setting up the configs
config = configparser.ConfigParser()
config.read('config.ini')
path_dataset=config['FOLDERS']['dataset']

### Reading the config datasets immidiately to allow for running different configs in parrallel.
if len(list(config['DATASETS'].keys()))>1:
    #Analysis.py can only handle one active dataset in config.ini!
    print("Please only specify one active dataset in config.ini for the analysis!")
    exit()
else:
    cfg=str(list(config['DATASETS'].keys())[0])+ ".ini"
    dataset_config = configparser.ConfigParser()
    dataset_config.read(os.path.join("cfg/datasets", cfg))

#Alternatively the different datasets can be looped from here.

#MetricHandler. Takes the independent configs as input
MetricHandler = metricUtils.Metrics(c, config,dataset_config, args.codec, args.debug, geometry)

#Loop structure
#### for all qualities:
####    for all point_cloud_sequences:
####        for all point_clouds in point_cloud_sequence:
####            #Calculate metrics

for qualityIni in args.quality:
    quality = qualityIni.split('.')[0]
    qualityInt = quality

    for pointCloudSequence in dataset_config['POINTCLOUDS']:
        print("Quality: ",quality," PointCloud_Sequence:",pointCloudSequence)
        #path handling.
        decomDatasetPath=os.path.join(config['FOLDERS']['decompressed'], args.codec, quality, dataset_config['BASE']['name'],
                                      pointCloudSequence,"attributes")
        inputDatasetPath = os.path.join(config['FOLDERS']['dataset'], dataset_config['BASE']['baseDir'], dataset_config['POINTCLOUDS'][pointCloudSequence])
        _, _, files = next(os.walk(decomDatasetPath))
        files.sort()

        #Iterating through the files in the dataset adding the values and averaging
        count=0
        for file in files:
            start=time.time()
            #Read point clouds
            decomFilePath=os.path.join(decomDatasetPath,file)
            inputFilePath=os.path.join(inputDatasetPath,file)
            pcd_com = o3d.io.read_point_cloud(decomFilePath)
            pcd_org = o3d.io.read_point_cloud(inputFilePath)

            #Calculate metrics
            MetricHandler.updateMetrics(pcd_org,pcd_com,pointCloudSequence,qualityInt)

            count += 1
            print("count ",count," Process time ",time.time()-start)
            #used to terminate for testing.
            #if count>=1:
            #    break

        #Print the metrics here for the given point cloud sequence.
        MetricHandler.printMetrics(pointCloudSequence,qualityInt, datasetName=dataset_config['BASE']['name'])
        #Option to save metrics each loop. Pickles are deprecated. #Only use this to save csv files of each quality.
        #MetricHandler.saveMetrics(dataset,qualityInt,pickle=False,csvfile=True)

        #Saves the logs
        MetricHandler.saveLog(pointCloudSequence,qualityInt)

#### Saving of the full metrics metrics ####
for pointCloudSequence in dataset_config['POINTCLOUDS']:
    ####Saving the metrics for all the qualities. Otherwise use the saveMetrics
    MetricHandler.saveAllMetricsCSV(pointCloudSequence, datasetName=dataset_config['BASE']['name'], quality = args.quality)
