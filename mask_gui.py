'''
Code inspired and adapted from Carsen Stringer CellPose https://github.com/MouseLand/cellpose/blob/master/cellpose/gui.py
'''

import sys, os, pathlib, warnings, datetime, tempfile, glob, time
import gc
from natsort import natsorted
from tqdm import tqdm

from PyQt5 import QtGui, QtCore, Qt, QtWidgets
import pyqtgraph as pg
from pyqtgraph import GraphicsScene

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

import sys

import h5py

import time

from matplotlib import image

from pyqtgraph import Point

class MainW(QtGui.QMainWindow):
    def __init__(self, image=None):
        super(MainW, self).__init__()

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 1200, 1000)

        self.setStyleSheet("QMainWindow {background: 'black';}")

        self.cwidget = QtGui.QWidget(self)
        self.l0 = QtGui.QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.setCentralWidget(self.cwidget)

        self.win = pg.GraphicsLayoutWidget()
        #self.l0.addWidget(self.win, 0,3, b, 20)

        self.l0.addWidget(self.win)


        #self.show()

        self.load_image()

        self.make_viewbox()

        self.set_image()

        self.win.show()

    def load_image(self):
        filename='C:/Users/koester_lab/Documents/Maria/registered/fish2_6dpf_medium_aligned.h5'
        filename_='C:/Users/koester_lab/Documents/im.jpg'
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            start=time.time()
            data=f['data'][100,10,:,:]
            end=time.time()
            print('Time to load file: ',end-start)
            print(data.shape)
        self.data=np.array(data).astype('float64')
        #self.data=image.imread(filename_)
        print(np.max(self.data))
        print(np.min(self.data))
        self.data *= 255.0/self.data.max()
        print(np.max(self.data))
        print(np.min(self.data))

    def set_image(self):
        self.img.setImage(self.data, autoLevels=False, lut=None,levels=[0,255])
        self.show()

    def make_viewbox(self):
        self.p0 = pg.ViewBox(invertY=True)
        self.brush_size=3
        self.win.addItem(self.p0, 0, 0)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = ImageDraw(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.p0.scene().contextMenuItem = self.p0
        self.p0.setMouseEnabled(x=False,y=False)
        self.Ly,self.Lx = 512,512
        self.p0.addItem(self.img)



class ImageDraw(pg.ImageItem):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    GraphicsObject displaying an image. Optimized for rapid update (ie video display).
    This item displays either a 2D numpy array (height, width) or
    a 3D array (height, width, RGBa). This array is optionally scaled (see
    :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
    with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
    before being displayed.
    ImageItem is frequently used in conjunction with
    :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
    :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
    for controlling the levels and lookup table used to display the image.
    """



    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        super(ImageDraw, self).__init__()

    def mouseDragEvent(self, ev):
        '''
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            print(ev.pos())
            return
        else:
            #ev.accept()
            print(ev.pos())
            self.drawAt(ev.pos(), ev)
        '''
        print(ev.pos())

    def mouseClickEvent(self, ev):
        print(ev.pos())



def run(image=None):
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QtGui.QApplication(sys.argv)

    main_window=MainW(image=image)
    main_window.show()
    ret = app.exec_()
    #sys.exit(ret)

run()
