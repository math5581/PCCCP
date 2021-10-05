import configparser
import argparse
import os
import subprocess
import codecLib
import numpy as np
from datetime import datetime
from shutil import copyfile
from visualizer import Visualizer

def makeDirIfNotExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

_,_,codecNames = next(os.walk("cfg/codecs"))
codecNames = [os.path.splitext(x)[0] for x in codecNames]

parser = argparse.ArgumentParser()
parser.add_argument('--codec', type=str, help='Codec to compress with [draco, gpcc, vpcc]')
parser.add_argument('--quality', type=str, help='Compression quality [names of configs, all (execute all configs)]')
parser.add_argument('--saveVideo', action="store_true", help="Save point cloud sequence to video, warning, is slow.")
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()
args.codec = args.codec.lower()
args.quality = [args.quality]

if args.debug:
    print("Debug on")

if args.codec not in codecNames:
    print("Not a valid codec")
    print("Valid codecs: ")
    [print("\t: ", x) for x in codecNames]
    exit()

_,_, qualityConfigs = next(os.walk(os.path.join("cfg/codecs", args.codec)))
qualityConfigs.sort()
qualityConfigs.append("all")

if args.quality[0] not in qualityConfigs:
    print(args.quality[0], " is an invalid --quality argument")
    print("Valid quality parameters: ")
    [print("\t: ", x) for x in qualityConfigs]
    exit()

if args.quality[0] == 'all':
    qualityConfigs.pop()
    args.quality = qualityConfigs

config = configparser.ConfigParser()
config.read('config.ini')

configPath = os.path.join("cfg/codecs/", args.codec + ".ini")
c = configparser.ConfigParser()
c.read(configPath)

if args.codec == "draco":
    codec = codecLib.Draco(c, args.debug)

elif args.codec == "gpcc" or args.codec == "gpcc-octree-lift" or args.codec == "gpcc-trisoup-lift" or args.codec == "gpcc-trisoup-raht":
    codec = codecLib.GPCC(c, args.debug)

elif args.codec == "vpcc":
    codec = codecLib.VPCC(c, args.debug)

for qualityIni in args.quality:
    quality = qualityIni.split('.')[0]
    qualityIniPath = os.path.join("cfg/codecs", args.codec, qualityIni)
    codec.setParameters(qualityIniPath)

    logFolder = os.path.join("output/", args.codec, quality) #output folder to store compression/decompression time logs in
    makeDirIfNotExist(os.path.join(config['FOLDERS']['compressed'], args.codec, quality))
    makeDirIfNotExist(os.path.join(config['FOLDERS']['decompressed'], args.codec, quality))
    makeDirIfNotExist(os.path.join("output/", args.codec, quality))
    copyfile(codec.activeCfg, os.path.join(logFolder, codec.activeCfg.split('/')[-1]))

    for cfg in config['DATASETS']:
        if cfg[-4:] != ".ini":
            cfg = cfg + ".ini"
        if args.debug == True:
            print("Quality config: ", codec.activeCfg, "\nDataset config: ", cfg)
            print()

        dataset = configparser.ConfigParser()
        dataset.read(os.path.join("cfg/datasets", cfg))

        comPath = os.path.join(config['FOLDERS']['compressed'], args.codec, quality, dataset['BASE']['name'])
        decomPath = os.path.join(config['FOLDERS']['decompressed'], args.codec, quality, dataset['BASE']['name'])
        makeDirIfNotExist(comPath)
        makeDirIfNotExist(decomPath)

        for key in dataset['POINTCLOUDS']:
            makeDirIfNotExist(os.path.join(comPath, key, "geometry"))
            makeDirIfNotExist(os.path.join(comPath, key, "attributes"))
            makeDirIfNotExist(os.path.join(decomPath, key, "geometry"))
            makeDirIfNotExist(os.path.join(decomPath, key, "attributes"))

            inputPath = os.path.join(config['FOLDERS']['dataset'], dataset['BASE']['baseDir'], dataset['POINTCLOUDS'][key])

            _,_,files = next(os.walk(inputPath))
            files.sort()

            print("Processing: ", dataset['BASE']['name'] + " - " + key)

            if args.debug == True:
                print("Located in: ", inputPath)

            decomTimeFile = os.path.join(logFolder, dataset['BASE']['name'] + " - " + key + "-decomtime.txt")
            comTimeFile = os.path.join(logFolder, dataset['BASE']['name'] + " - " + key + "-comtime.txt")

            # De/Compress without attributes, process time ignored
            _, compressedFiles = codec.encode(dataPathIn = inputPath, inFiles = files, dataPathOut=os.path.join(comPath,key, "geometry"), attributes = False)
            _ = codec.decode(inFiles = compressedFiles, dataPathOut=os.path.join(decomPath,key, "geometry"))

            # De/Compress with attributes
            comTime, compressedFiles = codec.encode(dataPathIn = inputPath, inFiles = files, dataPathOut=os.path.join(comPath,key, "attributes"), attributes = True)
            decomTime = codec.decode(inFiles = compressedFiles, dataPathOut=os.path.join(decomPath,key, "attributes"))

            comTime = np.array(comTime)
            decomTime = np.array(decomTime)
            np.savetxt(comTimeFile, comTime)
            np.savetxt(decomTimeFile, decomTime)

            if args.saveVideo:
                viz = Visualizer(path = os.path.join(decomPath,key), subsampleFactor = 1, out = os.path.join(logFolder, args.codec + "-" + qualityInt2Arg[qualityInt] + "-" + os.path.splitext(cfg)[0] + " - " + key + ".mp4"))
                viz.run()
