import configparser
import subprocess
import os

class codecBase():
    """ Base class for codecs """

    def __init__(self, cfg, debug = False):

        self.debug = debug
        self.cfg = cfg
        self.ext = self.cfg['BASE']['fileExt']
        self.encoder = "\""+self.cfg['BASE']['encoderPath']+"\""
        self.decoder = "\""+self.cfg['BASE']['decoderPath']+"\""
        self.cfgPath = os.path.join("cfg/codecs", self.cfg['BASE']['cfgFolder'])
        self.cfgList = []
        self.activeCfg = ""

        _,_, self.cfgList = next(os.walk(self.cfgPath))
        self.cfgList.sort()

        self.setParameters(os.path.join(self.cfgPath,self.cfgList[0]))


    def execute(self, pString):
        """ Executes a cmd/terminal command """
        if os.name == 'nt': ### Checking if windows
            result = subprocess.run(pString, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run([pString], shell=True, capture_output=True, text=True)
        return result

class Draco(codecBase):
    """docstring for Draco."""

    def __init__(self, cfg, debug = False):

        super().__init__(cfg, debug)

    def setParameters(self, iniFile):
        """ Sets the codec parameters based on a config file """

        cfg = configparser.ConfigParser()
        cfg.read(iniFile)
        self.activeCfg = iniFile

        self.quantization = str(cfg['QUALITY']['quantizationGeom'])  # quantizes point cloud to N bits, default 11 bits (2048x2048x2048 voxels)
        self.compression = str(cfg['QUALITY']['compressionLevel'])  # Draco's own compression parameter
        self.quantizationRGB = str(cfg['QUALITY']['quantizationRGB']) # quantizes point cloud RGB texture to N bits, default 8 bits
        print("\nDraco initialized with: \n", self.encoder, "\n", self.decoder, "\n\nActive configuration: ", self.activeCfg, "\nQuantization Geometry: ", self.quantization, "\nQuantization RGB: ", self.quantizationRGB, "\nCompression: ", self.compression, "\n")

    def encode(self, dataPathIn, inFiles, dataPathOut, attributes=False):
        """ compress a list of point clouds and saves it to disk
        input:
            - dataPathIn, list: Storage folder for uncompressed .ply files
            - inFiles, list: .ply files
            - dataPathOut, string: Folder to store compressed files
        output:
            - processTime, list: encoding times
            - out, list: Paths to compressed files
        """

        out, processTime = [], []
        for i in range(len(inFiles)):
            out.append(os.path.join(dataPathOut, os.path.splitext(inFiles[i])[0] + self.ext))
            file = "\"" + os.path.join(dataPathIn, inFiles[i]) + "\""

            p = self.encoder + " -point_cloud -i " + file + " -o " + out[i] + " -cl " + self.compression + " -qp " + self.quantization + " -qt " + self.quantizationRGB

            if not attributes:
                p += " --skip COLOR"

            if self.debug == True:
                print(p)
                print()

            result = self.execute(p)

            if self.debug == True:
                print(result.stdout)
                print()


            index=5
            if os.name == 'nt': ##checking if windows
                index=4
            s = result.stdout.split('\n')[index]
            processTime.append(int(s.split('(')[1].split(' ')[0]))

            print(i, "/", len(inFiles), ": \t encode Time: ", processTime[-1])

        return processTime, out

    def decode(self, inFiles, dataPathOut):
        """ Decompress a list of point clouds and saves it to disk in .ply format
        input:
            - inFiles, list: absolute path to compressed files
            - dataPathOut, string: Folder to store decompressed files
        output:
            - processTime, list: decoding times
        """

        processTime = []
        for i in range(len(inFiles)):
            outFile = os.path.join(dataPathOut, os.path.splitext(inFiles[i].split(os.path.sep)[-1])[0] + ".ply")
            p = self.decoder + " -i " + inFiles[i] + " -o " + outFile

            if self.debug == True:
                print(p)
                print()

            result = self.execute(p)

            if self.debug == True:
                print(result.stdout)
                print()

            processTime.append(int(result.stdout.split('(')[1].split(' ')[0]))
            print(i, "/", len(inFiles), ": \t decode Time: ", processTime[-1])

        return processTime

class GPCC(codecBase):
    """docstring for MPEG's GPCC."""

    def __init__(self, cfg, debug = False):

        super().__init__(cfg, debug)

    def setParameters(self, iniFile):
        """ Sets the codec parameters based on a config file """

        cfg = configparser.ConfigParser()
        cfg.read(iniFile)
        self.activeCfg = iniFile

        self.attributeBitDepth = str(cfg['QUALITY']['attrBitDepth'])  # GPCC's bit depth for attributes
        self.scaleFactor = str(cfg['QUALITY']['positionQuantizationScale']) # GPCC's scale factor used for changing geometry bit rates
        self.lumaQuantization = str(cfg['QUALITY']['lumaQuantization'])
        self.transformType = str(cfg['QUALITY']['transformType']) # 0: predictive (lossless attribute), 1: RAHT (lossy attribute) 2: lift (lossy attribute)
        self.trisoupNodeSize = str(cfg['QUALITY']['trisoupNodeSizeLog2'])
        print("\GPCC initialized with: \n", self.encoder, "\n", self.decoder, "\n\nActive configuration: ", self.activeCfg, "\nAttribut Bit Depth: ", self.attributeBitDepth, "\npositionQuantizationScale: ", self.scaleFactor, "\nLuma Quantization: ", self.lumaQuantization, "\n")

    def encode(self, dataPathIn, inFiles, dataPathOut, attributes=False):
        """ compress a list of point clouds and saves it to disk
        input:
            - dataPathIn, list: Storage folder for uncompressed .ply files
            - inFiles, list: .ply files
            - dataPathOut, string: Folder to store compressed files
        output:
            - processTime, list: encoding times
            - out, list: Paths to compressed files
        """

        out, processTime = [], []
        for i in range(len(inFiles)):
            out.append(os.path.join(dataPathOut, os.path.splitext(inFiles[i])[0] + self.ext))
            file = "\"" + os.path.join(dataPathIn, inFiles[i]) + "\""

            p = self.encoder + " --mode=0 --uncompressedDataPath=" + file + " --compressedStreamPath=" + out[-1] + " --positionQuantizationScale=" + self.scaleFactor + " --trisoupNodeSizeLog2=" + self.trisoupNodeSize + " --convertPlyColourspace=1 --transformType=" + self.transformType + " --numberOfNearestNeighborsInPrediction=3" + " --adaptivePredictionThreshold=64 --qp=" + self.lumaQuantization + " --qpChromaOffset=0 --bitdepth=" + self.attributeBitDepth

            if attributes:
                p += " --attribute=color"

            if self.debug == True:
                print(p)
                print()

            result = self.execute(p)

            if self.debug == True:
                print(result.stdout)
                print()

            s = result.stdout.split('\n')
            processTime.append(int(float(s[-2].split(' ')[-2])*1000))
            print(i + 1, "/", len(inFiles), ": \t encode Time: ", processTime[-1])

        return processTime, out

    def decode(self, inFiles, dataPathOut):
        """ Decompress a list of point clouds and saves it to disk in .ply format
        input:
            - inFiles, list: absolute path to compressed files
            - dataPathOut, string: Folder to store decompressed files
        output:
            - processTime, list: decoding times
        """

        processTime = []
        for i in range(len(inFiles)):
            outFile = os.path.join(dataPathOut, os.path.splitext(inFiles[i].split(os.path.sep)[-1])[0] + ".ply")
            p = self.encoder + " --mode=1 --compressedStreamPath=" + inFiles[i] + " --reconstructedDataPath=" + outFile
            result = self.execute(p)
            s = result.stdout.split('\n')
            processTime.append(int(float(s[-2].split(' ')[-2])*1000))
            print(i, "/", len(inFiles), ": \t decode Time: ", processTime[-1])

        return processTime

class VPCC(codecBase):
    """docstring for MPEG's VPCC."""

    def __init__(self, cfg, debug = False):

        super().__init__(cfg, debug)

    def setParameters(self, iniFile):
        """ Sets the codec parameters based on a config file """

        cfg = configparser.ConfigParser()
        cfg.read(iniFile)
        self.activeCfg = iniFile
        self.cfgPath = self.cfg['BASE']['cfgPath']
        self.qualityCfg = os.path.join(self.cfgPath, "rate", cfg['QUALITY']['cfg'])
        print("\nV-PCC initialized with: \n", self.encoder, "\n", self.decoder, "\n", self.cfgPath, "\n Quality cfg: ", self.qualityCfg, "\n")

    def encode(self, dataPathIn, inFiles, dataPathOut, attributes=False):
        """ compress a point cloud and save the bit stream to disk
        dataPathIn: Folder with PLY sequences
        inFiles: Files in dataPathIn ordered nummerically
        dataPathOut: Folder to store compressed .bin

        Returns: -list: Total processing time in ms
                - string: bit stream path
        """

        frameCount = str(len(inFiles))
        startFrame = 0
        file = ""

        i = 0
        for x in inFiles[0].split('_'):

            if i < len(inFiles[0].split('_')) -1:
                file += x + "_"
            else:
                startFrame = os.path.splitext(x)[0]
                x = "%04d.ply"
                fileWithOut = file
                file += x

            i += 1

        comOut = os.path.join(dataPathOut, os.path.splitext(inFiles[0])[0])
        ctcCommon = os.path.join(self.cfgPath, "common/ctc-common.cfg")
        ctcIntra = os.path.join(self.cfgPath, "condition/ctc-all-intra.cfg")

        uncompressedDataPath = os.path.join(dataPathIn, file)
        uncompressedDataPath = "\"" + uncompressedDataPath + "\""
        comOut = "\"" + comOut + self.ext + "\""

        if self.debug == True:
            print("Encode uncompressedDataPath: ", uncompressedDataPath, "\n output path ", comOut)
            print("Cfg 1: ", ctcCommon)
            print("Cfg 2: ", ctcIntra)
            print("Start frame number: ", startFrame)
            print("Number of frames: ", frameCount)

        p = self.encoder + " --uncompressedDataPath=" + uncompressedDataPath + " --frameCount=" + frameCount + " --compressedStreamPath=" + comOut + " --configurationFolder=" + self.cfgPath + " --config="+ctcCommon + " --config=" + ctcIntra + " --config=" + self.qualityCfg + " --startFrameNumber=" + startFrame + " --computeMetrics=0 --flagGeometrySmoothing=0"

        if not attributes:
            p += " --noAttributes"

        if self.debug == True:
            print("\n", p)

        result = self.execute(p)

        if self.debug == True:
            print("\n", result.stdout)

        s = result.stdout.split('\n')[-5]#.split(':')[-1].split(' ')[0]
        s = s.split(':')[-1]#.split(' ')[0]
        s = s.split(' ')[1]
        processTime = int(float(s)*1000)

        print("Encode Time: ", processTime)

        return [processTime], comOut

    def decode(self, inFiles, dataPathOut):
        """ Decompress a point cloud .bin and saves .ply files to a folder
        inFiles: compressed bit stream file
        dataPathOut: Folder to store decompressed .ply files

        Returns - list: process time in ms
        """

        startFrame = inFiles.replace('"', "").split('/')[-1][:-4].split('_')[-1]
        compressedStreamPath = inFiles
        outputFile = inFiles.replace('"', "").split('/')[-1][:-8] + "%04d.ply"
        reconstructedDataPath = os.path.join(dataPathOut, outputFile)
        cfgInverse = os.path.join(self.cfgPath, "hdrconvert/yuv420torgb444.cfg")

        if self.debug == True:
            print("Decode reconstructedDataPath: ", reconstructedDataPath, "\n compressed path ", compressedStreamPath)
            print("Cfg 1: ", cfgInverse)
            print("Start frame number: ", startFrame)

        p = self.decoder + " --compressedStreamPath=" + compressedStreamPath + " --inverseColorSpaceConversionConfig=" + cfgInverse + " --reconstructedDataPath=" + reconstructedDataPath + " --startFrameNumber=" + startFrame + " --computeMetrics=0 --computeChecksum=0"
        result = self.execute(p)

        if self.debug == True:
            print("\n", p, "\n", result.stdout)

        s = result.stdout.split('\n')[-5]
        s = s.split(':')[-1]
        s = s.split(' ')[1]
        processTime = int(float(s)*1000)

        print("Decode Time: ", processTime)

        return [processTime]
