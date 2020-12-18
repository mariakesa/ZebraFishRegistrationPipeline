from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from PyQt5 import QtCore, QtGui, QtWidgets


import time
import numpy as np
from vispy import app
import vispy

from PyQt5.QtWidgets import *
import vispy.app
import sys

from vispy.app import use_app
use_app('PyQt5')
from vispy import scene
from vispy import color
from vispy.color.colormap import Colormap

import h5py


import imageio
from vispy import visuals

class Canvas(scene.SceneCanvas):

    def __init__(self,rois_path,rm_path):
        scene.SceneCanvas.__init__(self,keys='interactive', size=(1024, 1024))

        self.unfreeze()

        self.plane_ind=0

        self.i=0

        self.rois_path=rois_path
        self.rm_path=rm_path

        self.load_data()

        self.view=self.central_widget.add_view()

        self.cm=color.get_colormap("nipy_spectral").map(self.rasterm)
        single_plane=self.pos[:,2]==self.plane_ind
        cm=self.cm[single_plane]
        colors=vispy.color.ColorArray(cm,alpha=0.8)
        Scatter2D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        self.p1 = Scatter2D(parent=self.view.scene)
        self.p1.set_data(self.rois_plane[:,:2], face_color=colors, symbol='o', size=6,
            edge_width=0.5, edge_color='blue')

        colormap = color.get_colormap("nipy_spectral")
        self.colorbar=scene.visuals.ColorBar(clim=(round(self.min,2),round(self.max,2)),cmap=colormap,orientation='right',size=(100,30),label_str='cl',parent=self.view.scene,pos=(100,100),label_color='white')

        self.colorbar.label.font_size = 10

    def load_data(self):
        self.pos=np.load(self.rois_path)
        single_plane=self.pos[:,2]==self.plane_ind
        self.rois_plane=self.pos[single_plane]

        self.rasterm=np.load(self.rm_path)
        self.rasterm_=self.rasterm[single_plane]

        self.min=np.min(self.rasterm)
        self.max=np.max(self.rasterm)

    def disp(self):
        single_plane=self.pos[:,2]==self.plane_ind
        self.rois_plane=self.pos[single_plane]
        self.rasterm_=self.rasterm[single_plane]
        #self.image.set_data(self.raw_data)
        cm=self.cm[single_plane]
        colors=vispy.color.ColorArray(cm,alpha=0.8)
        self.p1.set_data(self.rois_plane[:,:2], face_color=colors, symbol='o', size=6,
            edge_width=0.5, edge_color='blue')


class MainWindow(QMainWindow):
    def __init__(self, canvas=None,parent=None):
        super(MainWindow, self).__init__(parent)
        self.canvas=canvas
        widget = QWidget()
        self.setCentralWidget(widget)
        self.l0 = QGridLayout()
        self.l0.addWidget(canvas.native)
        widget.setLayout(self.l0)
        self.writer = imageio.get_writer('C:/Users/koester_lab/Documents/rastermap_2.gif')

        self.timer_init()

    def timer_init(self):
        self.timer = vispy.app.Timer()
        self.timer.connect(self.update)
        self.timer.start(0)
        self.timer.interval=0.2
        self.canvas.i=0

    def update(self,ev):
        self.canvas.plane_ind=self.canvas.i
        self.canvas.disp()
        self.canvas.i+=1
        im=self.canvas.render()
        self.writer.append_data(im)
        if self.canvas.i>=20:#canvas.raw_data.shape[0]:
            self.writer.close()
            import sys
            sys.exit()

def run_rm_viz(roi_path,rm_path):
    canvas = Canvas(roi_path,rm_path)
    vispy.use('PyQt5')
    w = MainWindow(canvas)
    w.show()
    vispy.app.run()
