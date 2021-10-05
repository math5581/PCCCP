import open3d as o3d
import numpy as np
import os
import time
import cv2
from tqdm import tqdm
import argparse
import os
import imageio
from skimage import img_as_ubyte
from visualizer import Visualizer

# Folder with ply files
path = "/media/albert/My_Passport/aivero/decompressed/gpcc/low/8iVFBv2/longdress"

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help="Folder with .ply sequence")
parser.add_argument('--subsampleFactor', action="store", type=int, help="Only visualize every N'th frame")
parser.add_argument('--out', action="store",type=str, help="Specify file name to save to mp4 file")
args = parser.parse_args()

if args.subsampleFactor == None:
    args.subsampleFactor = 1

viz = Visualizer(path = args.input, subsampleFactor = args.subsampleFactor, out = args.out)
viz.run()
