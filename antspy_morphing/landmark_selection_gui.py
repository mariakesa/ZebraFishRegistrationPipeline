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

class Canvas(scene.SceneCanvas):

    def __init__(self):
        scene.SceneCanvas.__init__(self,keys='interactive', size=(1024, 1024))
        self.unfreeze()
        self.i=0
        self.pos=np.array([[0,0]])
        self.colors=[0,0,0,1]
        self.index = 0
        #self.markers = visuals.MarkersVisual()
        #self.markers=scene.visuals.Markers(pos=pos, parent=wc_2.scene, face_color='blue')

        #self.markers.set_data(self.pos, face_color=self.colors)
        #self.markers.symbol = visuals.marker_types[10]
        #self.markers.transform = STTransform()
        self.plane_ind=0
        self.filename='//ZMN-HIVE/User-Data/Maria/Caiman_MC/fish11_6dpf_medium_aligned.h5'
        #self.filename='//ZMN-HIVE/User-Data/Maria/check_registration/control/fish11_6dpf_medium_aligned.h5'
        self.load_image()

        self.view=self.central_widget.add_view()

        self.image=scene.visuals.Image(self.im, parent=self.view.scene, cmap='hsv',clim=[0,255])
        self.image.set_gl_state('translucent', depth_test=False)
        self.markers=scene.visuals.Markers(pos=self.pos, parent=self.view.scene, face_color='blue')
        self.nrs=[]


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

    def make_nr(self):
        nr=scene.visuals.Text(str(self.i),color='blue',font_size=10,
            pos=self.pos[-1]+[20,-20],bold=True,parent=self.view.scene)
        self.nrs.append(nr)



    def print_mouse_event(self, event, what):
        """ print mouse events for debugging purposes """
        print('%s - pos: %r, button: %s,  delta: %r' %
              (what, event.pos, event.button, event.delta))
    def on_mouse_press(self, event):
        self.print_mouse_event(event, 'Mouse press')
        self.pos=np.vstack((self.pos,event.pos))
        print(self.pos)
        self.colors = np.vstack((self.colors,(153/255, 255/255, 255/255, 1)))
        print(self.colors)
        self.markers.set_data(self.pos, face_color=self.colors,size=15)
        self.make_nr()
        self.i+=1
        self.update()

class MainWindow(QMainWindow):
    def __init__(self, canvas=None,parent=None):
        super(MainWindow, self).__init__(parent)
        self.canvas=canvas
        widget = QWidget()
        self.setCentralWidget(widget)
        self.l0 = QGridLayout()
        self.l0.addWidget(self.canvas.native)
        self.slider_image = QSlider()
        self.slider_image.setOrientation(Qt.Horizontal)
        self.slider_image.setRange(0,20)
        self.slider_image.valueChanged.connect(self.slider_image_val_changed)
        self.l0.addWidget(self.slider_image)
        widget.setLayout(self.l0)

    def slider_image_val_changed(self):
        print('slider nr:', self.slider_image.tickPosition(),self.slider_image.value())



canvas = Canvas()
vispy.use('PyQt5')
w = MainWindow(canvas)
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
