import open3d as o3d
import math
import numpy as np
from sklearn.neighbors import KDTree
import time
import os
from collections import namedtuple
import scipy.interpolate
import pickle as pkl
import csv
import collections
import copy
import math


def getBBoxNorm(pcd):
    """ Returns the norm2 of the axis aligned bounding box """
    p = np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points())
    x = np.max(p[:,0]) - np.min(p[:,0])
    y = np.max(p[:,1]) - np.min(p[:,1])
    z = np.max(p[:,2]) - np.min(p[:,2])

    return math.sqrt(x**2 + y**2 + z**2)

def getPSNR(rmse, peakSignal):
    """ Returns the PSNR in decibel based on RMSE and the normalizing factor """
    try:
        psnr = 10*math.log10(peakSignal**2 / (rmse**2))
        return psnr
    except:
        return 0

def getNearestNeighbor(arr1, arr2, returnIndices=False):
    """ Input: np.array([x, y, z])
    output: distances np.array(distN)
            indices np.array(index) """

    X1 = KDTree(arr1, leaf_size=30, metric='euclidean')
    X2 = KDTree(arr2, leaf_size=30, metric='euclidean')

    distArr1, indices1 = X2.query(arr1, k=1)
    distArr2, indices2 = X1.query(arr2, k=1)

    if returnIndices == True:
        return distArr1, indices1.flatten(), distArr2, indices2.flatten()
    else:
        return distArr1, distArr2

def getD1Geom(pcd1, pcd2, type="float"):
    """ Returns the D1 PSNR of geometry """

    distArr1, distArr2 = getNearestNeighbor(np.asarray(pcd1.points), np.asarray(pcd2.points), returnIndices=False)
    rmse1 = np.sqrt(np.mean((distArr1)**2))
    rmse2 = np.sqrt(np.mean((distArr2)**2))

    rmse = max(rmse1, rmse2)

    if type == "float":
        peakSignal = getBBoxNorm(pcd2) # on the compressed point cloud
    elif type == "voxel":
        pcd_com = np.asarray(pcd2.points)
        peakSignal = max(np.max(pcd_com[:,0]), np.max(pcd_com[:,1]), np.max(pcd_com[:,2]))
    psnr = getPSNR(rmse, peakSignal)
    return psnr


def rgb2ycgcr(rgb):
    """ converts the rgb values to y cg cr according to ITU-R BT.709 conversion """
    mat = np.array([[0.2126, 0.7152, 0.0722],
                    [-0.1146, -0.3854, 0.5],
                    [0.5, -0.4542, -0.0458]])

    rgb = rgb.transpose()
    ycgcr = np.matmul(mat, rgb).transpose()
    #have changed the calculation of yuv. y is default a value [0,1], but cg and cr is [-0.5,0.5].
    #Hence the following makes the values of cg and cr between [0,1]
    ycgcr[:,1:]+=1
    temp=np.reshape(ycgcr[:,0],(ycgcr[:,0].shape[0],1))
    ycgcr=np.append(temp,ycgcr[:,1:]/2,axis=1)
    return ycgcr


def getD1Color(pcd1, pcd2):
    """ Returns the D1 PSNR of color (Y, Cb, Cr) """
    color1 = rgb2ycgcr(np.asarray(pcd1.colors)) * 255
    color2 = rgb2ycgcr(np.asarray(pcd2.colors)) * 255


    _, indices1, _, indices2 = getNearestNeighbor(np.asarray(pcd1.points), np.asarray(pcd2.points), returnIndices=True)

    indices1 = indices1.tolist()
    indices2 = indices2.tolist()

    distArr1 = color1 - color2[indices1]
    distArr2 = color2 - color1[indices2]
    psnr = []
    #calculates psnr for the three different color components.
    for i in range(3):
        #it needs to be the absolut value here, since the dist is a norm and cannot be negative.
        rmse1 = np.sqrt(np.mean(np.abs(distArr1[:,i])**2))
        rmse2 = np.sqrt(np.mean(np.abs(distArr2[:,i])**2))

        rmse = max(rmse1, rmse2)

        psnr.append(getPSNR(rmse, peakSignal=255)) # all color points are between 0 and 255

    return psnr[0], psnr[1], psnr[2] # psnr-Y, psnr-Cb, psnr-Cr

def getD1All(pcd1, pcd2,type="float",problem=False):
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    color1 = np.asarray(pcd1.colors)
    color2 = np.asarray(pcd2.colors)
    #remove the first point from color, if "problem" is true. Used to handle a GPCC reading bug.
    #This solution has the consequence of returning a very high
    if problem:
        points2 = np.asarray(pcd2.points)[1:]
        color2 = np.asarray(pcd2.colors)[1:]

    distArr1,indices1, distArr2,indices2 = getNearestNeighbor(points1, points2, returnIndices=True)

    rmse1_geom = np.sqrt(np.mean((distArr1)**2))
    rmse2_geom = np.sqrt(np.mean((distArr2)**2))
    rmse_geom = max(rmse1_geom, rmse2_geom)
    if type == "float":
        peakSignal_geom = getBBoxNorm(pcd2) # on the compressed point cloud
    elif type == "voxel":
        pcd_com = points2
        #peakSignal_geom = max(np.max(pcd_com[:,0]), np.max(pcd_com[:,1]), np.max(pcd_com[:,2]))
        ###### HAVE BEEN CHANGED TO THIS.
        peakSignal_geom = np.linalg.norm([np.max(pcd_com[:,0]),np.max(pcd_com[:,1]),np.max(pcd_com[:,2])])
    psnr_geom = getPSNR(rmse_geom, peakSignal_geom)

    color1 = rgb2ycgcr(color1) * 255
    color2 = rgb2ycgcr(color2) * 255

    indices1 = indices1.tolist()
    indices2 = indices2.tolist()
    distArr1 = color1 - color2[indices1]
    distArr2 = color2 - color1[indices2]
    psnr = []
    #calculates PSNR for the different color components
    for i in range(3):
        #it needs to be the absolut value here, since the dist is a norm and cannot be negative.
        rmse1 = np.sqrt(np.mean(np.abs(distArr1[:,i])**2))
        rmse2 = np.sqrt(np.mean(np.abs(distArr2[:,i])**2))

        rmse = max(rmse1, rmse2)

        psnr.append(getPSNR(rmse, peakSignal=255)) # all color points are between 0 and 255

    # Calculates the psnr for all color values at once.
    rmse1 = np.sqrt(np.mean(np.linalg.norm(distArr1,ord=2,axis=1)**2))
    rmse2 = np.sqrt(np.mean(np.linalg.norm(distArr2,ord=2,axis=1)**2))
    rmse = max(rmse1, rmse2)
    psnrCol=getPSNR(rmse, peakSignal=np.linalg.norm([255,255,255])) # all color points are between 0 and 255. Takes the norm of the diagnal.

    return psnr_geom,psnrCol,psnr[0], psnr[1], psnr[2] # psnr-Y, psnr-Cb, psnr-Cr

#Functions of bits per input point..
def getFileSize(pathCompressedFile,fileExtention=False):
    """ Returns filesize in bits """
    return os.path.getsize(str(pathCompressedFile[:-4]+fileExtention))*8

def getNumbPoints(pcdInput):
    """ Returns number of points in pcd """
    return np.asarray(pcdInput.points).shape[0]

def getBPP(pcdInput,pathCompressed,fileType):
    """ Returns Bits Per input Point """
    numbPoints=np.asarray(pcdInput.points).shape[0]
    sizeBits=getFileSize(pathCompressed,fileType)
    return sizeBits/numbPoints

def makeDirIfNotExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class Metrics:
    """ Metric class to keep track of the metrics for an independent codec and point cloud sequence. Initialized with:
        config_codec: config_codec file read from configPath = 'cfg/codecs/CODECTOUSE.ini'
        config: config file read from configPath = 'config.ini'
        dataset_config: confic file read from configPath = 'cfg/datasets/DATASETTOUSE.ini'
        codec: speficy codec #Can possibly be obtained from config_codec
        debug: True/False
        geometry: For internal use. Used not to calculate metrics for geometry only.
    """
    def __init__(self,config_codec,config,dataset_config,codec,debug,geometry):
        #debug, geometry and codec params.
        self.geometry = geometry
        self.codec = codec
        self.debug=debug
        #Initializing the configs here.
        self.dataset_config = dataset_config
        self.config = config
        self.compressedBasePath = config['FOLDERS']['compressed']
        self.datasetBasePath = config['FOLDERS']['dataset']
        if type(config_codec) is str:
            print("Warning, config for codec is not specified!")
        else:
            self.fileType=config_codec['BASE']['fileExt']

        #Metric setups.
        self.metrics = {"psnrGeom":[],"psnrCol":[],"psnrY":[],"psnrCb":[],
                        "psnrCr":[],"numpoints":0,"count":0}

        #Convertions of qualities
        self.qualityArg2Int = {'lowest': 0, 'low': 1, 'medium': 2, 'best': 3, 'all': 4}
        self.qualityInt2Arg = {0: 'lowest', 1: 'low', 2: 'medium', 3: 'best', 4: 'all'}

        #Storaged of the calculated metrics.
        self.metricDict={}

        #Storage of the finished calculated metrics.
        self.__outputMetricDict={}

        #Deprecated, previously used to replace the inf values to calculate BD metrics.
        #Max for the independent PSNRs. Meant to be universal and used to replace with infinite values...
        self.maxPSNR={"G":70,"Y":45,"Cb":55,"Cr":45}

        #### Character to replace inf values in output.
        self.replace="lossless"

    def __updateMetricDict(self,pointCloudSequence,compressionRateInt):
        """Private function to handle the Metric directories"""
        if pointCloudSequence not in self.metricDict:
            self.metricDict[pointCloudSequence]={}
        if compressionRateInt not in self.metricDict[pointCloudSequence]:
            self.metricDict[pointCloudSequence][compressionRateInt]=copy.deepcopy(self.metrics)#deepcopy the memory placement of the metrics. Otherwise it overwrites.

    def updateMetrics(self,pcd_ori,pcd_deg,pointCloudSequence,compressionRateInt):
        """Updates all the metrics for """
        self.__updateMetricDict(pointCloudSequence,compressionRateInt)
        # Used to handle the gpcc reading bug.
        problem=False
        if self.codec == "gpcc":
            problem=True

        #possibly adjust the voxel here.
        temp_psnrGeom,temp_psnrColor,temp_pnsry,temp_psnrcb,temp_psnrcr = getD1All(pcd_ori,pcd_deg, type='voxel',problem=problem)
        #update psnrGeom
        self.metricDict[pointCloudSequence][compressionRateInt]['psnrGeom'].append(temp_psnrGeom)
        #update psnrColor
        self.metricDict[pointCloudSequence][compressionRateInt]['psnrCol'].append(temp_psnrColor)
        self.metricDict[pointCloudSequence][compressionRateInt]['psnrY'].append(temp_pnsry)
        self.metricDict[pointCloudSequence][compressionRateInt]['psnrCb'].append(temp_psnrcb)
        self.metricDict[pointCloudSequence][compressionRateInt]['psnrCr'].append(temp_psnrcr)

        #Update full number of points in point cloud sequence
        num_points=getNumbPoints(pcd_ori)
        self.metricDict[pointCloudSequence][compressionRateInt]['numpoints'] += num_points

        self.metricDict[pointCloudSequence][compressionRateInt]['count'] +=1
        if self.debug:
            print("Dataset:\t",pointCloudSequence)
            print("PSNR Geometry[dB]:\t",temp_psnrGeom)
            print("PSNR Color[dB]:\t\t ",temp_psnrColor)
            print("PSNR Y[dB]:\t\t ",temp_pnsry)
            print("PSNR Cb[dB]:\t\t ",temp_psnrcb)
            print("PSNR Cr[dB]:\t\t ",temp_psnrcr)


    def getAveragePSNRGeom(self,pointCloudSequence,compressionRateInt):
        """ Returns the average PSNR for Geometry """
        return np.sum(self.metricDict[pointCloudSequence][compressionRateInt]['psnrGeom'])/self.metricDict[pointCloudSequence][compressionRateInt]['count']

    def getAveragePSNRCol(self,pointCloudSequence,compressionRateInt):
        """ Returns the average PSNR for Color """
        return np.sum(self.metricDict[pointCloudSequence][compressionRateInt]['psnrCol'])/self.metricDict[pointCloudSequence][compressionRateInt]['count']

    def getAveragePSNRY(self,pointCloudSequence,compressionRateInt):
        """ Returns the average PSNR for Y"""
        return np.sum(self.metricDict[pointCloudSequence][compressionRateInt]['psnrY'])/self.metricDict[pointCloudSequence][compressionRateInt]['count']

    def getAveragePSNRCb(self,pointCloudSequence,compressionRateInt):
        """ Returns the average PSNR for Cb"""
        return np.sum(self.metricDict[pointCloudSequence][compressionRateInt]['psnrCb'])/self.metricDict[pointCloudSequence][compressionRateInt]['count']

    def getAveragePSNRCr(self,pointCloudSequence,compressionRateInt):
        """ Returns the average PSNR for Cr"""
        return np.sum(self.metricDict[pointCloudSequence][compressionRateInt]['psnrCr'])/self.metricDict[pointCloudSequence][compressionRateInt]['count']

    #Only works, if it is looped through the entire dataset
    def getAverageBPPGeom(self,pointCloudSequence,compressionRateInt,compDatasetPath):
        """ Returns the average Bit Per Point Value for geometry """
        compDatasetPath = os.path.join(compDatasetPath,"geometry")
        compressedSize = self.getFolderSize(compDatasetPath)

        return compressedSize/self.metricDict[pointCloudSequence][compressionRateInt]['numpoints']

    #Only works, if it is looped through the entire dataset
    def getAverageBPPAttr(self,pointCloudSequence,compressionRateInt,compDatasetPath):
        """ Returns the average Bit Per Point Value"""
        compDatasetPath = os.path.join(compDatasetPath,"attributes")
        compressedSize = self.getFolderSize(compDatasetPath)
        return compressedSize/self.metricDict[pointCloudSequence][compressionRateInt]['numpoints']


    def getNumberOfFiles(self,pathFiles):
        """ Returns number of files in a point cloud sequence """
        return int(len(next(os.walk(pathFiles))[2]))

    def getFolderSize(self,start_path):
        """ Returns the folder size in bits"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)

        return total_size*8

    def getCompressionRatioGeom(self,inputDatasetPath,compressedDataPath):
        """ Returns compression ratio in % """
        compressedDataPath = os.path.join(compressedDataPath,"geometry")
        compressedSize = self.getFolderSize(compressedDataPath)
        datasetSize = self.getFolderSize(inputDatasetPath)
        return compressedSize/datasetSize

    def getCompressionRatioFull(self,inputDatasetPath,compressedDataPath):
        """Returns compression ratio in %"""
        compressedDataPath = os.path.join(compressedDataPath,"attributes")
        compressedSize = self.getFolderSize(compressedDataPath)
        datasetSize = self.getFolderSize(inputDatasetPath)
        return compressedSize/datasetSize

    def getAverageEncodingTime(self,qualityInt, pathDataset, datasetName):
        """Returns the average encoding times for specified qualityInt and point cloud sequence"""
        pathToOutputCodecs = os.path.join("output", self.codec, qualityInt)
        files = next(os.walk(pathToOutputCodecs))[2]
        numberOfFiles = self.getNumberOfFiles(pathDataset)
        for file in files:
            if file[-12:] == "-comtime.txt" and datasetName in file:
                path = os.path.join(pathToOutputCodecs, file)
                arr = np.loadtxt(path).astype(np.float)
                averageEncodingTime = arr.sum()/numberOfFiles
        return averageEncodingTime

    def getAverageDecodingTime(self,qualityInt, pathDataset, datasetName):
        """Returns the average decoding times for specified qualityInt and point cloud sequence"""
        pathToOutputCodecs = os.path.join("output", self.codec, qualityInt)
        files = next(os.walk(pathToOutputCodecs))[2]
        numberOfFiles = self.getNumberOfFiles(pathDataset)
        for file in files:
            if file[-14:] == "-decomtime.txt" and datasetName in file:
                path = os.path.join(pathToOutputCodecs, file)
                arr = np.loadtxt(path).astype(np.float)
                averagedecodingTime = arr.sum()/numberOfFiles
        return averagedecodingTime

    # Used to store the finished metrics calculated.
    def __updateOutputMetricDict(self, pointCloudSequence, compressionRateInt):
        """Private function to handle the Metric directories"""
        # This needs to take in the given compression rate.
        if pointCloudSequence not in self.__outputMetricDict:
            self.__outputMetricDict[pointCloudSequence] = {}  # self.metric
        if compressionRateInt not in self.__outputMetricDict[pointCloudSequence]:
            self.__outputMetricDict[pointCloudSequence][compressionRateInt] = {}

    # Used as an interface to access the finished calcualted metrics.
    def __constructOutputMetricsDict(self, pointCloudSequence, qualityInt, datasetName):
        self.__updateOutputMetricDict(pointCloudSequence, qualityInt)
        #Makes this a bit nicer... Integrate with the different datasets and quality.
        # BPP
        compDataPath = os.path.join(self.compressedBasePath,self.codec,qualityInt,self.dataset_config['BASE']['baseDir'],pointCloudSequence)
        pointCloudSequencePath = os.path.join(self.datasetBasePath,self.dataset_config['BASE']['baseDir'],self.dataset_config['POINTCLOUDS'][pointCloudSequence])

        if self.geometry:
            self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_BPP_Geom"] = self.getAverageBPPGeom(pointCloudSequence, qualityInt,compDataPath)
        self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_BPP_Attr"] = self.getAverageBPPAttr(pointCloudSequence, qualityInt,compDataPath)

        # Update PSNR
        self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_psnrGeom"] = self.getAveragePSNRGeom(pointCloudSequence, qualityInt)
        self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_psnrCol"] = self.getAveragePSNRCol(pointCloudSequence, qualityInt)
        self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_psnrY"] = self.getAveragePSNRY(pointCloudSequence, qualityInt)
        self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_psnrCb"] = self.getAveragePSNRCb(pointCloudSequence, qualityInt)
        self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_psnrCr"] = self.getAveragePSNRCr(pointCloudSequence, qualityInt)

        # Update CompressionRatio
        if self.geometry:
            self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_compressionRatio_Geom"] = self.getCompressionRatioGeom(pointCloudSequencePath,compDataPath)
        self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_compressionRatio_Full"] = self.getCompressionRatioFull(pointCloudSequencePath,compDataPath)

        # Update enc/dec time
        self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_enc_time"] = self.getAverageEncodingTime(qualityInt, pointCloudSequencePath, datasetName)
        self.__outputMetricDict[pointCloudSequence][qualityInt]["avg_dec_time"] = self.getAverageDecodingTime(qualityInt, pointCloudSequencePath, datasetName)

        # Update count.
        self.__outputMetricDict[pointCloudSequence][qualityInt]["count"] = self.metricDict[pointCloudSequence][qualityInt]["count"]



    def printMetrics(self,pointCloudSequence,qualityInt, datasetName):
        """ Prints the evaluations metrics for the specified point cloud sequence and quality parameter. """
        self.__constructOutputMetricsDict(pointCloudSequence,qualityInt, datasetName)
        print("Printing metrics  for ","point cloud sequence: ", pointCloudSequence, " Quality: ", qualityInt)
        for key in self.__outputMetricDict[pointCloudSequence][qualityInt]:
            print(key," ",self.__outputMetricDict[pointCloudSequence][qualityInt][key])

    #Used to save metrics if not looped through the entire qualities.
    def saveMetrics(self, pointCloudSequence, qualityInt, datasetName, pickle=True,csvfile=True):
        """Saves the evaluation metrics for the specified point cloud sequence and quality"""
        outputFolder="metrics"
        makeDirIfNotExist(outputFolder)
        # get the metrics here for the given pointCloudSequence.
        self.__constructOutputMetricsDict(pointCloudSequence, qualityInt, datasetName)

        #Pickles are deprecated.
        if pickle:
            print("Saving pickle file for: ","pointCloudSequence: ", pointCloudSequence, " Quality: ", self.qualityInt2Arg[qualityInt])
            path=os.path.join(outputFolder,"pkl")
            makeDirIfNotExist(path)
            str_save=str(self.codec)+"_"+str(pointCloudSequence)+"_"+str(self.qualityInt2Arg[qualityInt])
            path=os.path.join(path,str_save)
            output = open(path+'.pkl', 'wb')
            pkl.dump(self.__outputMetricDict, output)
            output.close()
        if csvfile:
            print("Saving csv file for: ","pointCloudSequence: ", pointCloudSequence, " Quality: ", self.qualityInt2Arg[qualityInt])
            path=os.path.join(outputFolder,"csv",self.dataset_config["BASE"]["name"],pointCloudSequence)
            makeDirIfNotExist(path)
            str_save=str(self.codec)+"_"+str(pointCloudSequence)+"_"+str(self.qualityInt2Arg[qualityInt])
            path = os.path.join(path, str_save)
            with open(path+".csv", "w+") as csv_file:
                writer = csv.writer(csv_file, dialect='excel', lineterminator = '\n',
                                    quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(self.__outputMetricDict[pointCloudSequence][qualityInt].keys())
                temp = self.replaceInf(list(self.__outputMetricDict[pointCloudSequence][qualityInt].values()))
                writer.writerow(temp)

    def saveLog(self,pointCloudSequence,qualityInt):
        """ Saves the log files for PSNR values for the specified quality int and point cloud sequence."""
        outputFolder="metrics"
        outputFolder=os.path.join(outputFolder,"logs")
        outputFolder=os.path.join(outputFolder,self.dataset_config["BASE"]["name"],pointCloudSequence)
        makeDirIfNotExist(outputFolder)
        print("Saving log file for: ", "point cloud sequence: ", pointCloudSequence, " Quality: ", qualityInt)
        str_save = str(self.codec) + "_" + qualityInt + "_log"
        path = os.path.join(outputFolder, str_save)
        with open(path + ".csv", "w+") as csv_file:
            writer = csv.writer(csv_file, dialect='excel', lineterminator='\n',
                                quoting=csv.QUOTE_NONNUMERIC)
            # Writing Header
            temp=list(self.metricDict[pointCloudSequence][qualityInt].keys())
            writer.writerow(temp[:5]+["count"])
            self.metricDict[pointCloudSequence][qualityInt].keys()
            data=list(self.metricDict[pointCloudSequence][qualityInt].values())
            # Writing the PSNR values.
            for i in range(self.metricDict[pointCloudSequence][qualityInt]['count']):
                PSNR=[data[0][i],data[1][i],data[2][i],data[3][i],data[4][i]]
                temp=PSNR+[i]
                writer.writerow(temp)

    def saveAllMetricsCSV(self, pointCloudSequence, datasetName, quality):
        """ Saves the evaluation metrics for all the specified specified point cloud sequence.
            Can only save as csv. Requires a loop through all the 4 quality ints. """
        outputFolder="metrics"
        # get the metrics here for the given point cloud sequence. Possibly change to get the 4 from somewhere else...
        for qualityIni in quality:
            #construct for them all:
            qualityStr = qualityIni.split('.')[0]
            self.__constructOutputMetricsDict(pointCloudSequence, qualityStr, datasetName)
        print("Saving csv file for: ","point cloud sequence: ", pointCloudSequence)
        path=os.path.join(outputFolder,"csv",self.dataset_config["BASE"]["name"],pointCloudSequence)
        makeDirIfNotExist(path)

        str_save=str(self.codec)+"_"+str(pointCloudSequence)+"_metrics"
        path = os.path.join(path, str_save)

        with open(path+".csv", "w+") as csv_file:
            writer = csv.writer(csv_file, dialect='excel', lineterminator = '\n',
                                quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["quality"]+list(self.__outputMetricDict[pointCloudSequence][quality[0].split('.')[0]].keys()))
            for qualityIni in quality:
                #construct for them all:
                qualityStr = qualityIni.split('.')[0]
                temp=self.replaceInf(list(self.__outputMetricDict[pointCloudSequence][qualityStr].values()))
                writer.writerow([qualityStr]+temp)

    def replaceInf(self,lis):
        """ Replace the inf values of a list with a predefined value """
        for i in range(len(lis)):
            if math.isinf(lis[i]):
                lis[i]=self.replace
        return lis

    def calculateBDPSNR(self,bitRates1,PSNRValues1,bitRates2,PSNRValues2):
        """Calculates the Bjøntegaard Delta PSNR for two bit rates and corresponding PSNR values. Requires at least 4 values for each b"""
        """Adopted from: https://github.com/Anserw/Bjontegaard_metric """
        bitRates1 = np.asarray(bitRates1)
        bitRates2 = np.asarray(bitRates2)

        lR1=np.log(bitRates1)
        lR2=np.log(bitRates2)

        PSNR1=np.asarray(PSNRValues1)
        PSNR2=np.asarray(PSNRValues2)

        p1=np.polyfit(lR1,PSNR1,3)
        p2=np.polyfit(lR2,PSNR2,3)

        min_int=max(min(lR1),min(lR2))
        max_int=min(max(lR1),max(lR2))

        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        # Think this can be more precise the higher num chosen here. 100 seems fine though
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]

        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)

        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

        # find avg diff
        avg_diff = (int2 - int1) / (max_int - min_int)

        return avg_diff

    def calculateBDRate(self,bitRates1,PSNRValues1,bitRates2,PSNRValues2):
        """Calculates the Bjøntegaard Delta Rate for two bit rates and corresponding PSNR values. Requires at least 4 values."""
        """Adopted from: https://github.com/Anserw/Bjontegaard_metric """
        bitRates1 = np.asarray(bitRates1)
        bitRates2 = np.asarray(bitRates2)

        lR1=np.log(bitRates1)
        lR2=np.log(bitRates2)

        PSNR1=np.asarray(PSNRValues1)
        PSNR2=np.asarray(PSNRValues2)

        p1=np.polyfit(lR1,PSNR1,3)
        p2=np.polyfit(lR2,PSNR2,3)

        min_int=max(min(lR1),min(lR2))
        max_int=min(max(lR1),max(lR2))

        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        #Think this can be more precise the higher num chosen here. 100 seems fine though
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)

        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

        # find avg diff
        avg_exp_diff = (int2 - int1) / (max_int - min_int)
        avg_diff = (np.exp(avg_exp_diff) - 1) * 100
        return avg_diff
