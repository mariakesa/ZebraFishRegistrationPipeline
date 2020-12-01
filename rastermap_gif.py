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




        #self.image=scene.visuals.Image(self.raw_data,parent=self.view.scene, cmap='grays',clim=[0,255])
        #self.image.set_gl_state('translucent', depth_test=False)
        self.cm=color.get_colormap("nipy_spectral").map(self.rasterm)
        single_plane=self.pos[:,2]==self.plane_ind
        cm=self.cm[single_plane]
        colors=vispy.color.ColorArray(cm,alpha=0.8)
        Scatter2D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        self.p1 = Scatter2D(parent=self.view.scene)
        self.p1.set_data(self.rois_plane[:,:2], face_color=colors, symbol='o', size=6,
            edge_width=0.5, edge_color='blue')

        colormap = color.get_colormap("nipy_spectral")
        self.colorbar=scene.visuals.ColorBar(clim=(round(self.min,2),round(self.max,2)),cmap=colormap,orientation='right',size=(100,30),label_str='spc',parent=self.view.scene,pos=(100,100),label_color='white')

        self.colorbar.label.font_size = 10
        #self.colorbar.draw()
        #self.time=0

        #self.time_text=scene.visuals.Text(str(time),color='white',pos=(100,0),bold=True)




    def create_cell_image(self):
        self.cell_act=np.zeros((1024,1024))
        self.cell_act[self.rois_plane[:,0],self.rois_plane[:,1]]=1
        self.cell_act=np.rot90(self.cell_act,3)

    def load_data(self):
        filename='C:/Users/koester_lab/Documents/Maria/registered/fish17_6dpf_medium_aligned_andreas.h5'
        filename='//ZMN-HIVE/User-Data/Maria/masked/fish20_6dpf_medium_masked_detrended_.h5'
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Loading raw data from a plane...")
            start=time.time()
            self.raw_data=f['data'][0,self.plane_ind,:,:].astype('float32')
            end=time.time()
            print('Time to load raw data file: ',end-start)
        self.raw_data*= 0.1

        pos_file='//ZMN-HIVE/User-Data/Maria/segmented/fish20_6dpf_medium_masked_detrended__rois.npy'
        self.pos=np.load(pos_file)
        #self.time_s=np.load('Y:/Maria/segmented/fish2_6dpf_medium_detrended_traces.npy')
        #self.pos=np.load('Y:/Maria/segmented/fish2_6dpf_medium_detrended_rois.npy')
        single_plane=self.pos[:,2]==self.plane_ind
        self.rois_plane=self.pos[single_plane]

        self.rasterm=np.load('rasterm.npy')
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
        self.pl_ind_box=QLineEdit()
        self.pl_ind_box.setText("0")
        self.pl_ind_box.setFixedWidth(35)
        self.l0.addWidget(self.pl_ind_box, 0, 4, 1, 2)
        self.pl_ind_box.returnPressed.connect(lambda: self.change_plane_ind())
        widget.setLayout(self.l0)
        self.writer = imageio.get_writer('C:/Users/koester_lab/Documents/rastermap.gif')

        self.timer_init()

    def timer_init(self):
        self.timer = vispy.app.Timer()
        self.timer.connect(self.update)
        self.timer.start(0)
        self.timer.interval=0.2
        canvas.i=0

    def update(self,ev):
        canvas.plane_ind=canvas.i
        canvas.disp()
        canvas.i+=1
        im=canvas.render()
        self.writer.append_data(im)
        if canvas.i>=20:#canvas.raw_data.shape[0]:
            self.writer.close()
            import sys
            sys.exit()

    def change_plane_ind(self):
        canvas.plane_ind=int(self.pl_ind_box.text())
        canvas.load_data()
        canvas.disp()
        #self.timer_init()

canvas = Canvas()
vispy.use('PyQt5')
w = MainWindow(canvas)
w.show()
vispy.app.run()
#vispy.app.processEvents(
