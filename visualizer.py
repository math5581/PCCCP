import open3d as o3d
import numpy as np
import os
import cv2
from tqdm import tqdm
import sys
import os
import imageio
from skimage import img_as_ubyte
import meshio

class Visualizer():
    """docstring for Visualizer."""

    def __init__(self, path, subsampleFactor, out=None):
        """ input:
                - path, str: folder with decompressed .ply files
                - out, str: filePath to save .mp4
        """
        self.path = path
        self.out = out
        self.subsampleFactor = subsampleFactor

        if self.out != None:
            self.fileOut = self.out
            if os.path.splitext(self.fileOut)[-1] != ".mp4":
                self.fileOut = os.path.splitext(self.fileOut)[0] + ".mp4"
            #Update for windows.
            if os.name == "nt":
                self.out_video = imageio.get_writer(self.fileOut, fps=30)
                print("Saving video to: ", self.fileOut)
            else:
                self.out_video = imageio.get_writer(self.fileOut, fps=30, macro_block_size = None)
                print("Saving video to: ", self.fileOut)

        # Set open3d verbosity level to surpress terminal/console output
        v_level = o3d.utility.VerbosityLevel.Warning
        o3d.utility.set_verbosity_level(v_level)

        self.viz = o3d.visualization.Visualizer()

    def getFilesNames(self):
        """ Returns the .ply file names of the point cloud sequence
            input:
                - subsampleFactor, int: >1, only take every N frame of sequence
        """
        _,_, files = next(os.walk(self.path))
        files.sort() # will probably have to update this sorting function call
        files = files[::self.subsampleFactor]
        return files

    def run(self):
        """ runs the visualization
            input:
                - subsampleFactor, int: >1, only take every N frame of sequence
        """

        print("Processing video...")
        self.viz.create_window()

        files = self.getFilesNames()

        for file in tqdm(files):

            # Had to do a stupid work around with meshio, since open3d couldn't
            # open .ply files compressed/decompressed with gpcc. Though this
            # solution is outcommented for now.

            ply = os.path.join(self.path, file)
            pcd = o3d.io.read_point_cloud(ply)

            """
            pcd = meshio.read(ply)
            xyz = pcd.points
            rgb = np.transpose(np.vstack((pcd.point_data['red'], pcd.point_data['green'], pcd.point_data['blue'])))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb/255.0)
            """

            self.viz.clear_geometries()
            self.viz.add_geometry(pcd)
            self.viz.update_geometry(pcd)
            self.viz.poll_events()
            self.viz.update_renderer()

            img = np.asarray(self.viz.capture_screen_float_buffer())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #cv2.imshow("renderer", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if self.out != None:
                self.out_video.append_data(img_as_ubyte(img[:, :, ::-1]))

        if self.out != None:
            self.out_video.close()
