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

    def __init__(self):
        scene.SceneCanvas.__init__(self,keys='interactive', size=(1024, 1024))

        self.unfreeze()

        self.plane_ind=0

        self.i=0

        self.load_data()

        self.view=self.central_widget.add_view()

        cm=color.get_colormap("cool").map(self.time_s_colors[:,0])

        self.image=scene.visuals.Image(self.raw_data[0,:,:],parent=self.view.scene, cmap='grays',clim=[0,255])
        self.image.set_gl_state('translucent', depth_test=False)

        colors=vispy.color.ColorArray(cm,alpha=0.8)
        Scatter2D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        self.p1 = Scatter2D(parent=self.view.scene)
        self.p1.set_data(self.rois_plane[:,:2], face_color=colors, symbol='o', size=8,
            edge_width=0.5, edge_color='blue')


    def create_cell_image(self):
        self.cell_act=np.zeros((1024,1024))
        self.cell_act[self.rois_plane[:,0],self.rois_plane[:,1]]=1
        self.cell_act=np.rot90(self.cell_act,3)

    def load_data(self):
        filename='C:/Users/koester_lab/Documents/Maria/registered/fish2_6dpf_medium_aligned.h5'
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Loading raw data from a plane...")
            start=time.time()
            self.raw_data=f['data'][:,self.plane_ind,:,:].astype('float32')
            end=time.time()
            print('Time to load raw data file: ',end-start)
        for j in range(0,self.raw_data.shape[0]):
            self.raw_data[j,:,:] *= 400.0/(self.raw_data[j,:,:].max()+0.00001)

        self.time_s=np.load('C:/Users/koester_lab/Documents/Maria/segmented/fish2_6dpf_medium_masked_traces.npy')
        self.pos=np.load('C:/Users/koester_lab/Documents/Maria/segmented/fish2_6dpf_medium_masked_rois.npy')
        single_plane=self.pos[:,2]==self.plane_ind
        self.rois_plane=self.pos[single_plane]
        self.time_s=self.time_s[single_plane]
        self.time_s_colors=self.time_s/800


class MainWindow(QMainWindow):
    def __init__(self, canvas=None,parent=None):
        super(MainWindow, self).__init__(parent)
        self.canvas=canvas
        widget = QWidget()
        self.setCentralWidget(widget)
        self.l0 = QGridLayout()
        self.l0.addWidget(canvas.native)
        self.pl_ind_box=QLineEdit()
        self.pl_ind_box.setText("0")
        self.pl_ind_box.setFixedWidth(35)
        self.l0.addWidget(self.pl_ind_box, 0, 4, 1, 2)
        self.pl_ind_box.returnPressed.connect(lambda: self.change_plane_ind())
        widget.setLayout(self.l0)

        self.timer_init()

    def timer_init(self):
        self.timer = vispy.app.Timer()
        self.timer.connect(self.update)
        self.timer.start(0)
        self.timer.interval=0.1

    def update(self,ev):
        cm=color.get_colormap("cool").map(canvas.time_s_colors[:,canvas.i])
        colors=vispy.color.ColorArray(cm,alpha=0.8)
        canvas.p1.set_data(canvas.rois_plane[:,:2], face_color=colors, symbol='o', size=8,
            edge_width=0.5, edge_color='blue')
        canvas.image.set_data(canvas.raw_data[canvas.i,:,:])
        print(canvas.i)
        canvas.i+=1
        if canvas.i>=canvas.raw_data.shape[0]:
            canvas.i=0

    def change_plane_ind(self):
        canvas.plane_ind=int(self.pl_ind_box.text())
        canvas.load_data()
        self.timer_init()

canvas = Canvas()
vispy.use('PyQt5')
w = MainWindow(canvas)
w.show()
vispy.app.run()
#vispy.app.processEvents(
