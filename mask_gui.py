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

        #Data and output arrays
        self.mask_arr=np.zeros((21,1024,1024)).astype('float64')


        #Plane ind
        self.plane_ind=0

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

        self.make_plane_input()

        self.make_plane_text_box()

        self.set_image()

        self.win.show()

    def load_image(self):
        inc_ind=0
        n_planes=21
        filename='C:/Users/koester_lab/Documents/Maria/registered/fish2_6dpf_medium_aligned.h5'
        filename_='C:/Users/koester_lab/Documents/im.jpg'
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            start=time.time()
            data=f['data'][inc_ind,:,:,:]
            end=time.time()
            print('Time to load file: ',end-start)
        self.data=np.array(data).astype('float64')
        for j in range(0,21):
            self.data[j,:,:] *= 255.0/self.data[j,:,:].max()
        self.data_masked=self.data.copy()

    def set_image(self):
        self.img.setImage(self.data[0,:,:], autoLevels=False, lut=None,levels=[0,255])
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
        self.segment_button=QtGui.QPushButton('Segment')
        #self.segment_button.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(self.segment_button, 1, 0,2,2)
        self.segment_button.clicked.connect(lambda: self.segment())

    def make_plane_input(self):
        self.plane_input=QtGui.QLineEdit(self)
        self.plane_input.setText("0")
        self.plane_input.setFixedWidth(35)
        self.plane_input.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.plane_input.returnPressed.connect(lambda: self.set_plane_ind_image())
        self.l0.addWidget(self.plane_input, 0, 0,2,1)

    def set_plane_ind_image(self):
        self.plane_ind=int(self.plane_input.text())
        self.img.setImage(self.data[self.plane_ind,:,:], autoLevels=False, lut=None,levels=[0,255])
        self.show()


    def segment(self):

        x=np.arange(0,1024)
        y=np.arange(0,1024)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))

        path = matplotlib.path.Path(self.img.pt_lst)
        mask = path.contains_points(points)
        mask=mask.reshape(1024,1024).astype('float64')

        self.data_masked[self.plane_ind,:,:][mask==0]=0

        self.mask_arr[self.plane_ind]=mask

        self.img.setImage(self.data_masked[self.plane_ind,:,:],autoLevels=False, lut=None,levels=[0,255])

    def make_plane_text_box(self):
        self.plane_text = QtGui.QTextEdit()
        cursor = self.plane_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText('>>>ERROR<<<\n')
        cursor.insertText('Hello')
        print(self.plane_text.toPlainText())
        self.plane_text.ensureCursorVisible()
        self.l0.addWidget(self.plane_text, 3,0,5,5)

    def update_text_box(self):
        pass



class ImageDraw(pg.ImageItem):

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
