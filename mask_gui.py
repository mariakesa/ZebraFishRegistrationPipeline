'''
Code inspired and adapted from Carsen Stringer CellPose https://github.com/MouseLand/cellpose/blob/master/cellpose/gui.py

https://stackoverflow.com/questions/50847827/how-can-i-select-the-pixels-that-fall-within-a-contour-in-an-image-represented-b
https://stackoverflow.com/questions/31542843/inpolygon-for-python-examples-of-matplotlib-path-path-contains-points-method
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
import matplotlib.path

from pyqtgraph import Point
from collections.abc import Callable

class MainW(QtGui.QMainWindow):
    def __init__(self, image=None):
        super(MainW, self).__init__()

        pg.setConfigOptions(imageAxisOrder="col-major")
        self.setGeometry(50, 50, 1200, 1000)

        self.setStyleSheet("QMainWindow {background: 'black';}")

        self.cwidget = QtGui.QWidget(self)
        self.l0 = QtGui.QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.setCentralWidget(self.cwidget)

        self.win = pg.GraphicsLayoutWidget()
        #self.l0.addWidget(self.win, 0,3, b, 20)

        self.l0.addWidget(self.win,0,10)


        #self.show()

        self.load_image()

        self.make_viewbox()

        self.make_buttons()

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
        self.win.addItem(self.p0, 3, 0)
        self.p0.setMenuEnabled(False)
        #self.p0.setMouseEnabled(x=True, y=True)
        self.p0.setAspectLocked(True)
        self.img = ImageDraw(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        #self.p0.scene().contextMenuItem = self.p0
        #self.p0.setMouseEnabled(x=False,y=False)
        kern = 100*np.ones((10,10))
        self.img.setDrawKernel(kern, mask=kern, center=(1,1), mode='add')
        self.p0.addItem(self.img)

    def make_buttons(self):
        b=0
        self.segment_button=QtGui.QPushButton('Segment')
        self.l0.addWidget(self.segment_button, b, 0,1,1)
        self.segment_button.clicked.connect(lambda: self.segment())

    def segment(self):
        print(self.img.pt_lst)

        x=np.arange(0,1024)
        y=np.arange(0,1024)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))

        path = matplotlib.path.Path(self.img.pt_lst)
        mask = path.contains_points(points)
        mask=mask.reshape(1024,1024)

        print(mask.shape)
        self.data[mask==False]=0
        self.img.setImage(self.data,autoLevels=False, lut=None,levels=[0,1])



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

        self.autoDownsample = False
        self.axisOrder = 'row-major'
        self.removable = False

        self.parent = parent
        #kernel[1,1] = 1
        #self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.pt_lst=[]



    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        elif self.drawKernel is not None:
            #print(ev.pos())
            ev.accept()
            self.drawAt(ev.pos(), ev)
            self.pt_lst.append([ev.pos().x(),ev.pos().y()])


    def drawAt(self, pos, ev=None):
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]

        for i in [0,1]:
            dx1 = -min(0, tx[i])
            dx2 = min(0, self.image.shape[0]-tx[i])
            tx[i] += dx1+dx2
            sx[i] += dx1+dx2

            dy1 = -min(0, ty[i])
            dy2 = min(0, self.image.shape[1]-ty[i])
            ty[i] += dy1+dy2
            sy[i] += dy1+dy2

        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        mask = self.drawMask
        src = dk

        if isinstance(self.drawMode, Callable):
            self.drawMode(dk, self.image, mask, ss, ts, ev)
        else:
            src = src[ss]
            if self.drawMode == 'set':
                if mask is not None:
                    mask = mask[ss]
                    self.image[ts] = self.image[ts] * (1-mask) + src * mask
                else:
                    self.image[ts] = src
            elif self.drawMode == 'add':
                self.image[ts] += src
            else:
                raise Exception("Unknown draw mode '%s'" % self.drawMode)
            self.updateImage()

    #def mouseClickEvent(self, ev):
        #print(ev.pos())





def run(image=None):
    # Always start by initializing Qt (only once per application)
    #warnings.filterwarnings("ignore")
    app = QtGui.QApplication(sys.argv)

    main_window=MainW(image=image)
    main_window.show()
    ret = app.exec_()
    #sys.exit(ret)

run()
