# PCCCP Evaluation Pipeline

Can run compression and decompression on the given .ply datasets.

Can evaluate the performance of the given codecs on the evaluation metrics given in the report.

## Codecs currently supported:

* Google's Draco
* MPEG's GPCC
* MPEG's VPCC

## Datasets currently supported:

* i8VFB-V2
* Robot@Home (subset)

## How to setup:
### Codecs

* Build and make Install Google's Draco: https://github.com/google/draco
* Build and make MPEG's G-PCC (tm13): https://github.com/MPEGGroup/mpeg-pcc-tmc13
* Build and make MPEG's G-PCC (tm2): https://github.com/MPEGGroup/mpeg-pcc-tmc2
* For each .ini file in "cfg/codecs" folder, edit the following:
  * encoderPath = /absolute/path/to/encoder/binary
  * decoderPath = /absolute/path/to/decoder/binary

### Dataset

* Edit config.ini. It contains the storage paths and the datasets you want to compress/decompress
  * dataset=/absolute/path/to/dataset/folder/
  * compressed=/absolute/path/to/compressed/point/clouds/folder/
  * decompressed=/absolute/path/to/decompressed/point/clouds/folder/

* Download the following datasets to the dataset folder given in config.ini
  * i8VFB-v2: http://plenodb.jpeg.org/pc/8ilabs
  * robotathome: https://ananas.isa.uma.es:10002/sharing/8SxEpuX9o

## How to run:

---

**Compression and decompression**
```
python3 run.py --codec [draco, gpcc, vpcc] --quality [lowest, low, medium, best, all]

[optional]
--saveVideo saves a video to output/codec/quality/.mp4 it is quite slow to run
--debug (python3 run.py --codec CODEC_HERE --quality QUALITY --debug)
```

Output compression time and decompression time in "output/CODEC_HERE/QUALITY/*.txt" as well as the active quality config file (codec.ini)
Outputs compressed and decompressed files as given in the config.ini file.

---

**Performance evaluation**
```
python3 analysis.py --codec [draco, gpcc, vpcc] --quality [lowest, low, medium, best, all] 
[optional]
--debug (python3 analysis.py --codec CODEC_HERE --quality QUALITY --debug)
```
Has been setup in similar style as run.py with arguments for the different compression methods.
The analysis file contains an object of the Metric class, which keeps track of the independent metrics for the different datasets. Updated once each iteration.
Calculates PSNR for Geometry, Y, Cb, Cr, Bitrates, BPP, compression and decompression time. Metrics can be saved to a 'metric' folder after the full iterations or retrieved while iterating through the different datasets. Log files are currently saved after each quality loop.

---

**Visualize Point Cloud Sequence**
```
python3 visualize.py --input pointCloudFolder --out [OPTIONAL] saveTo.mp4 --subsampleFactor [OPTIONAL] int
```
WARNING, this is quite slow

Takes a folder with a point cloud sequence in .ply format as input
Can save it as an mp4 file.
Can chose to only visualize every N'th frame

---

## To do:

- Add more datasets
- Create a script that downloads the datasets
- Finish the evaluation pipeline


## How to add a dataset?

Each dataset needs a .ini file in the "cfg/datasets/" folder. This .ini file contains the name of the base directory within the "dataset" folder given in "config.ini". Then it contains a list of point cloud sequence to be processed. Each point cloud sequence must be in their own directory with nothing else in it, should be in .PLY format.
Supports x (float), y (float), z (float), R (uchar), G (uchar), B (uchar).
x, y and z must be voxelzied integers.

See "cfg/datasets/i8vfbv2.ini" for a documented example.

Add the dataset to the "config.ini" file under the [DATASETS] section.

Folder structure:


  ├── datasetFolder   # Folder specified in config.ini </br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├──── baseDir     # Folder specified in "cfg/datasets/datasetToBeAdded.ini" </br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├───── pointcloudSequenceFolder1 # contains only .PLY files  </br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├───── pointcloudSequenceFolder2 # contains only .PLY files  </br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ├───── pointcloudSequenceFolder3 # contains only .PLY files  </br>

## How to configure compression codec?

Each codec has a .ini file located in "cfg/codecs/codecName", their performance can be adjusted by changing the parameters in the [QUALITY] section.
