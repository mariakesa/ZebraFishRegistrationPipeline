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

        self.max_time=self.raw_data.shape[0]

        self.fit_tensors()

        self.factors_to_tensors()

        self.view=self.central_widget.add_view()

        self.image=scene.visuals.Image(self.tensor_data[0,:,:],parent=self.view.scene, cmap='bwr',clim=[0,255])
        #self.image.set_gl_state('translucent', depth_test=False)

        self.time_text=scene.visuals.Text(str(self.i),color='white',font_size=25,pos=(900,100),bold=True,parent=self.view.scene)

    def fit_tensors(self):
        start=time.time()
        print('Fitting Tucker decomposition...')
        core, factors = tucker(self.raw_data, ranks = [20,50,50])
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
            self.raw_data=f['data'][:100,self.plane_ind,:,:].astype('float32')
            end=time.time()
            print('Time to load raw data file: ',end-start)
        #for j in range(0,self.raw_data.shape[0]):
            #self.raw_data[j,:,:] *= 200.0/(self.raw_data[j,:,:].max()+0.00001)




class MainWindow(QMainWindow):
    def __init__(self, canvas=None,parent=None):
        super(MainWindow, self).__init__(parent)
        self.create_canvi()
        self.canvas_lst_for_grid()
        self.l0 = QGridLayout()
        self.add_screens()
        self.t_c=QLineEdit()
        self.t_c.setText("0")
        self.t_c.setFixedWidth(35)
        self.l0.addWidget(self.t_c, 0, 14, 1, 2)
        self.t_c.returnPressed.connect(lambda: self.change_t_c())
        box = QGroupBox()
        box.setLayout(self.l0)
        self.setCentralWidget(box)

        #self.writer = imageio.get_writer('C:/Users/koester_lab/Documents/masked.gif')

        self.timer_init()

    def create_canvi(self):
        self.canvas_lst=[]
        self.file_paths=['//ZMN-HIVE/User-Data/Maria/Registered/fish2_6dpf_medium_aligned_andreas.h5','//ZMN-HIVE/User-Data/Maria/Registered/fish17_6dpf_medium_aligned_andreas.h5','//ZMN-HIVE/User-Data/Maria/Registered/fish9_6dpf_medium_aligned_andreas.h5','//ZMN-HIVE/User-Data/Maria/Registered/fish04_6dpf_amph_aligned_andreas.h5','//ZMN-HIVE/User-Data/Maria/Registered/fish41_6dpf_amph_aligned_andreas.h5','//ZMN-HIVE/User-Data/Maria/Registered/fish43_6dpf_amph_aligned_andreas.h5']
        for c in self.file_paths:
            self.canvas_lst.append(Canvas(c))
        min_times_interm=[]
        for canvas in self.canvas_lst:
            min_times_interm.append(canvas.max_time)
        self.min_time=min(min_times_interm)

    def canvas_lst_for_grid(self):
        self.canvas_grid=[]
        i=0
        for v in range(0,2):
            interm=[]
            for h in range(0,3):
                interm.append(self.canvas_lst[i+h])
            i=3
            self.canvas_grid.append(interm)
        print('canvas grid: ,',self.canvas_grid)

    def add_screens(self):
        i=0
        for v in range(0,2):
            z=0
            for h in range(0,3):
                self.l0.addWidget(self.canvas_grid[v][h].native,i+4,z+6,4,6)
                i=4
                z+=4
            #cntr+=4


    def timer_init(self):
        self.timer = vispy.app.Timer()
        self.timer.connect(self.update)
        self.timer.start(0)
        self.timer.interval=0.1
        for canvas in self.canvas_lst:
            canvas.i=0

    def update(self,ev):
        for canvas in self.canvas_lst:
            canvas.image.set_data(canvas.tensor_data[canvas.i,:,:])
            #print(canvas.raw_data[canvas.i,:,:])
            canvas.time_text.text=str(canvas.i)
            canvas.i+=1

            #im=canvas.render()
            #self.writer.append_data(im)
            if canvas.i>=self.min_time:
                #self.writer.close()
                #import sys
                #sys.exit()
                canvas.i=0

            canvas.update()

    def change_t_c(self):
        for canvas in self.canvas_lst:
            canvas.temporal_component=int(self.t_c.text())
            canvas.factors_to_tensors()
        self.timer_init()

#canvas = Canvas()
vispy.use('PyQt5')
app = QApplication(sys.argv)
w = MainWindow()
w.show()
vispy.app.run()
#vispy.app.processEvents()
