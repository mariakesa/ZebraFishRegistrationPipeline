#https://stackoverflow.com/questions/42007434/slider-widget-for-pyqtgraph

import numpy as np

from vispy import app, visuals

import h5py

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

from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider
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

import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt

from vispy.visuals.transforms import STTransform

from vispy.visuals.filters import Alpha

from PIL import Image
from scipy import ndimage

class Canvas(scene.SceneCanvas):

    def __init__(self,filename,type):
        scene.SceneCanvas.__init__(self, keys='interactive')
        self.unfreeze()
        self.type=type
        if self.type=='moving':
            self.size=(1024,1406)
        elif self.type=='atlas':
            self.size=(621,1406)
        self.i=0
        self.pos=np.array([[0,0]])
        self.colors=[0,0,0,1]
        self.index = 0

        self.pos_dict={}
        self.nrs_dict={}
        self.colors_dict={}

        if self.type=='moving':
            self.n_planes=21
        elif self.type=='atlas':
            self.n_planes=138

        for j in range(0,self.n_planes):
            self.pos_dict[j]=np.array([[0,0]])
            self.nrs_dict[j]=[]
            self.colors_dict[j]=[0,0,0,1]

        self.plane_ind=0
        self.filename=filename
        #self.filename='//ZMN-HIVE/User-Data/Maria/check_registration/control/fish11_6dpf_medium_aligned.h5'
        self.view=self.central_widget.add_view()
        self.markers_dict={}
        for j in range(0,self.n_planes):
            self.markers_dict[j]=scene.visuals.Markers(pos=self.pos_dict[self.plane_ind], parent=self.view.scene, face_color=self.colors_dict[self.plane_ind])
            #self.markers_dict[j].attach(Alpha(1))
            #transform = STTransform(translate=[0,0,-100])
            #self.markers_dict[j].transform = transform
        #transform = STTransform(translate=[0,0,-100])
        #self.markers.transform = transform
        self.nrs=[]
        if self.type=='moving':
            self.load_image()
        if self.type=='atlas':
            self.load_entire_tif()
            self.load_tif()

        #self.view=self.central_widget.add_view()

        self.image=scene.visuals.Image(self.im, parent=self.view.scene, cmap='hsv',clim=[0,255])
        self.image.set_gl_state('translucent', depth_test=False)
        self.image.attach(Alpha(0.4))
        #self.markers=scene.visuals.Markers(pos=self.pos, parent=self.view.scene, face_color='blue')
        #self.nrs=[]


    def load_image(self):
        with h5py.File(self.filename, "r") as f:
            # List all groups
            print("Loading raw data from a plane...")
            start=time.time()
            self.im=f['data'][0,self.plane_ind,:,:].astype('float32')
            self.im*= 400.0/(self.im.max()+0.00001)
            end=time.time()
            print('Time to load raw data file: ',end-start)
            print(np.max(self.im))
            #self.im[self.im>255]=200
            #self.im=np.array(self.im)
            #print(np.max(self.im))
            #print(np.min(self.im))

    def load_entire_tif(self):
        dataset = Image.open(self.filename)
        h,w = np.shape(dataset)
        tiffarray = np.zeros((h,w,dataset.n_frames))
        for i in range(dataset.n_frames):
           dataset.seek(i)
           tiffarray[:,:,i] = np.array(dataset)
        self.expim = tiffarray.astype(np.double)
        print(self.expim.shape)

    def load_tif(self):
        self.im=self.expim[:,:,self.plane_ind]
        self.im = ndimage.rotate(self.im, 270, reshape=True)
        self.im=self.im/20
        #self.im = ndimage.rotate(self.im, 270, reshape=True)
        #print(self.im)
        #print(self.im.shape)


    def make_nr(self):
        nr=scene.visuals.Text(str(self.i),color='blue',font_size=10,
            pos=self.pos_dict[self.plane_ind][-1]+[20,-20],bold=True,parent=self.view.scene)
        self.nrs_dict[self.plane_ind].append(nr)



    def print_mouse_event(self, event, what):
        """ print mouse events for debugging purposes """
        print('%s - pos: %r, button: %s,  delta: %r' %
              (what, event.pos, event.button, event.delta))
    def on_mouse_press(self, event):
        self.print_mouse_event(event, 'Mouse press')
        self.pos_dict[self.plane_ind]=np.vstack((self.pos_dict[self.plane_ind],event.pos))
        print('boom',self.pos_dict[self.plane_ind])
        print(self.plane_ind)
        self.colors_dict[self.plane_ind] = np.vstack((self.colors_dict[self.plane_ind],(153/255, 255/255, 255/255, 1)))
        print(self.colors_dict[self.plane_ind])
        self.markers_dict[self.plane_ind].set_data(self.pos_dict[self.plane_ind], face_color=self.colors_dict[self.plane_ind],size=15)
        print(self.pos_dict[self.plane_ind])
        print(self.markers_dict[self.plane_ind])
        self.make_nr()
        self.i+=1
        print(self.i)
        self.update()

class MainWindow(QMainWindow):
    def __init__(self, canvas_image=None,canvas_atlas=None,parent=None):
        super(MainWindow, self).__init__(parent)
        self.canvas_atlas=canvas_atlas
        self.canvas_image=canvas_image
        widget = QWidget()
        self.setCentralWidget(widget)
        self.l0 = QGridLayout()
        self.l0.addWidget(self.canvas_atlas.native,0,0,20,20)
        self.l0.addWidget(self.canvas_image.native,0,20,20,20)
        self.slider_atlas = QSlider()
        self.slider_atlas.setOrientation(Qt.Horizontal)
        self.slider_atlas.setRange(0,137)
        self.slider_atlas.valueChanged.connect(self.slider_atlas_val_changed)
        self.l0.addWidget(self.slider_atlas,20,0,1,20)
        self.slider_image = QSlider()
        self.slider_image.setOrientation(Qt.Horizontal)
        self.slider_image.setRange(0,20)
        self.slider_image.valueChanged.connect(self.slider_image_val_changed)
        self.l0.addWidget(self.slider_image,20,20,1,20)
        widget.setLayout(self.l0)
        self.prev_ind_image=0
        self.prev_ind_atlas=0

    def slider_image_val_changed(self):
        print('slider nr:', self.slider_image.tickPosition(),self.slider_image.value())
        self.canvas_image.markers_dict[self.prev_ind_image].visible=False
        for nr in self.canvas_image.nrs_dict[self.prev_ind_image]:
            nr.visible=False
        self.canvas_image.plane_ind=self.slider_image.value()
        self.canvas_image.load_image()
        self.canvas_image.image.set_data(self.canvas_image.im)
        self.canvas_image.image.set_gl_state('translucent', depth_test=False)
        self.canvas_image.markers_dict[self.canvas_image.plane_ind].set_data(self.canvas_image.pos_dict[self.canvas_image.plane_ind], face_color=self.canvas_image.colors_dict[self.canvas_image.plane_ind],size=15)
        self.canvas_image.markers_dict[self.canvas_image.plane_ind].visible=True
        for nr in self.canvas_image.nrs_dict[self.canvas_image.plane_ind]:
            nr.visible=True
        self.prev_ind_image=self.slider_image.value()
        self.canvas_image.update()

    def slider_atlas_val_changed(self):
        print('slider nr:', self.slider_atlas.tickPosition(),self.slider_atlas.value())
        self.canvas_atlas.markers_dict[self.prev_ind_atlas].visible=False
        for nr in self.canvas_atlas.nrs_dict[self.prev_ind_atlas]:
            nr.visible=False
        self.canvas_atlas.plane_ind=self.slider_atlas.value()
        self.canvas_atlas.load_tif()
        self.canvas_atlas.image.set_data(self.canvas_atlas.im)
        self.canvas_atlas.image.set_gl_state('translucent', depth_test=False)
        self.canvas_atlas.markers_dict[self.canvas_atlas.plane_ind].set_data(self.canvas_atlas.pos_dict[self.canvas_atlas.plane_ind], face_color=self.canvas_atlas.colors_dict[self.canvas_atlas.plane_ind],size=15)
        self.canvas_atlas.markers_dict[self.canvas_atlas.plane_ind].visible=True
        for nr in self.canvas_atlas.nrs_dict[self.canvas_atlas.plane_ind]:
            nr.visible=True
        self.prev_ind_atlas=self.slider_atlas.value()
        self.canvas_image.update()


canvas_image = Canvas(filename='//ZMN-HIVE/User-Data/Maria/Caiman_MC/fish11_6dpf_medium_aligned.h5',type='moving')
canvas_atlas = Canvas(filename='C:/Users/koester_lab/Documents/Maria/ZebraFishRegistrationPipeline/Elavl3-H2BRFP.tif',type='atlas')
vispy.use('PyQt5')
w = MainWindow(canvas_image,canvas_atlas)
w.show()
vispy.app.run()

'''
def load_image():
    filename='//ZMN-HIVE/User-Data/Maria/Caiman_MC/fish11_6dpf_medium_aligned.h5'
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Loading raw data from a plane...")
        start=time.time()
        im=f['data'][0,0,:,:].astype('float32')
        im*= 400.0/(im.max()+0.00001)
        end=time.time()
        print('Time to load raw data file: ',end-start)
    plt.imshow(im)
    plt.show()
load_image()
'''
