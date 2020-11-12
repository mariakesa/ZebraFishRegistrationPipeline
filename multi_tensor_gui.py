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

import tensorly as tl
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor

class Canvas(scene.SceneCanvas):

    def __init__(self,file_path):
        scene.SceneCanvas.__init__(self, keys='interactive', size=(1024, 1024))

        self.unfreeze()

        self.plane_ind=10

        self.temporal_component=0

        self.i=0

        self.filename=file_path
        self.load_data()

        self.fit_tensors()

        self.factors_to_tensors()

        self.view=self.central_widget.add_view()

        self.image=scene.visuals.Image(self.tensor_data[0,:,:],parent=self.view.scene, cmap='bwr',clim=[0,255])
        #self.image.set_gl_state('translucent', depth_test=False)

        self.time_text=scene.visuals.Text(str(self.i),color='white',font_size=25,pos=(900,100),bold=True,parent=self.view.scene)

    def fit_tensors(self):
        start=time.time()
        print('Fitting Tucker decomposition...')
        core, factors = tucker(self.raw_data, ranks = [100,50,50])
        end=time.time()
        print('Tucker done in:', end-start)
        self.core=core
        self.factors=factors

    def factors_to_tensors(self):
        #factors_=tl.tensor([self.factors[0][:,self.temporal_component].reshape(-1,1),self.factors[1],self.factors[2]])
        factors_=tl.tensor([self.factors[0][:,:self.temporal_component*5+1].reshape(-1,self.temporal_component*5+1),self.factors[1],self.factors[2]])
        print(factors_.shape)
        print(self.core.shape)
        self.tensor_data=tucker_to_tensor((self.core[:self.temporal_component*5+1,:,:].reshape(-1,50,50), factors_))
        for j in range(0,self.tensor_data.shape[0]):
            self.tensor_data[j,:,:] *= 300.0/(self.tensor_data[j,:,:].max()+0.00001)

    def load_data(self):
        #filename='C:/Users/koester_lab/Documents/Maria/registered/fish2_6dpf_medium_aligned_andreas.h5'
        #filename='//ZMN-HIVE/User-Data/Maria/Registered/fish2_6dpf_medium_aligned_andreas.h5'
        #filename='C:/Users/koester_lab/Documents/Maria/registered/fish9_6dpf_medium_aligned_andreas.h5'
        #filename='Y:/Maria/detrended/fish2_6dpf_medium_detrended.h5'
        #filename='C:/Users/koester_lab/Documents/Maria/masked/fish2_6dpf_medium_masked.h5'
        #filename='Y:/Maria/Registered/fish9_6dpf_amph_aligned_andreas.h5'
        #filename='//ZMN-HIVE/User-Data/Maria/Registered/fish43_6dpf_amph_aligned_andreas.h5'
        #filename='//ZMN-HIVE/User-Data/Maria/Registered/fish9_6dpf_medium_aligned_andreas.h5'
        #filename='C:/Users/koester_lab/Documents/Maria/registered/fish04_6dpf_amph_aligned_carsen.h5'
        with h5py.File(self.filename, "r") as f:
            # List all groups
            print("Loading raw data from a plane...")
            start=time.time()
            self.raw_data=f['data'][:,self.plane_ind,:,:].astype('float32')
            end=time.time()
            print('Time to load raw data file: ',end-start)
        #for j in range(0,self.raw_data.shape[0]):
            #self.raw_data[j,:,:] *= 200.0/(self.raw_data[j,:,:].max()+0.00001)




class MainWindow(QMainWindow):
    def __init__(self, canvas=None,parent=None):
        super(MainWindow, self).__init__(parent)
        self.create_canvi()
        widget = QWidget()
        self.setCentralWidget(widget)
        self.l0 = QGridLayout()
        self.l0.addWidget(canvas.native)
        self.t_c=QLineEdit()
        self.t_c.setText("0")
        self.t_c.setFixedWidth(35)
        self.l0.addWidget(self.t_c, 0, 8, 1, 2)
        self.t_c.returnPressed.connect(lambda: self.change_t_c())
        box = QGroupBox()
        box.setLayout(layout)
        self.setCentralWidget(box)
        widget.setLayout(box)

        #self.writer = imageio.get_writer('C:/Users/koester_lab/Documents/masked.gif')

        self.timer_init()

    def create_canvi(self):
        self.canvas_lst=[]
        self.file_paths=['//ZMN-HIVE/User-Data/Maria/Registered/fish9_6dpf_medium_aligned_andreas.h5','//ZMN-HIVE/User-Data/Maria/Registered/fish43_6dpf_amph_aligned_andreas.h5']
        for c in self.file_paths:
            self.canvas_lst.append(Canvas(c))

    def timer_init(self):
        self.timer = vispy.app.Timer()
        self.timer.connect(self.update)
        self.timer.start(0)
        self.timer.interval=0.1
        canvas.i=0

    def update(self,ev):
        canvas.image.set_data(canvas.tensor_data[canvas.i,:,:])
        #print(canvas.raw_data[canvas.i,:,:])
        print(canvas.i)
        canvas.time_text.text=str(canvas.i)
        canvas.i+=1

        #im=canvas.render()
        #self.writer.append_data(im)
        if canvas.i>=canvas.raw_data.shape[0]:
            #self.writer.close()
            #import sys
            #sys.exit()
            canvas.i=0

        canvas.update()

    def change_t_c(self):
        canvas.temporal_component=int(self.t_c.text())
        canvas.factors_to_tensors()
        self.timer_init()

#canvas = Canvas()
vispy.use('PyQt5')
app = QApplication(sys.argv)
w = MainWindow()
w.show()
vispy.app.run()
#vispy.app.processEvents(
