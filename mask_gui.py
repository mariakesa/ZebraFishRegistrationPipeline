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
        self.p0 = ViewBoxNoRightDrag(
            parent=self,
            lockAspect=True,
            name="plot1",
            border=[100, 100, 100],
            invertY=True
        )
        self.brush_size=3
        self.win.addItem(self.p0, 0, 0)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.layer = ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0,255])
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0,255])
        self.p0.scene().contextMenuItem = self.p0
        #self.p0.setMouseEnabled(x=False,y=False)
        self.Ly,self.Lx = 512,512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)


class ViewBoxNoRightDrag(pg.ViewBox):
    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True, invertY=False, enableMenu=True, name=None, invertX=False):
        pg.ViewBox.__init__(self, None, border, lockAspect, enableMouse,
                            invertY, enableMenu, name, invertX)
        self.parent = parent
        self.axHistoryPointer = -1

    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        The following events are implemented:
        +/= : moves forward in the zooming stack (if it exists)
        - : moves backward in the zooming stack (if it exists)
        """
        ev.accept()
        if ev.text() == '-':
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ['+', '=']:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()

    def mouseDragEvent(self, ev, axis=None):
        ## if axis is specified, event will only affect that axis.
        if self.parent is None or (self.parent is not None and not self.parent.in_stroke):
            ev.accept()  ## we accept all buttons

            pos = ev.pos()
            lastPos = ev.lastPos()
            dif = pos - lastPos
            dif = dif * -1

            ## Ignore axes if mouse is disabled
            mouseEnabled = np.array(self.state['mouseEnabled'], dtype=np.float)
            mask = mouseEnabled.copy()
            if axis is not None:
                mask[1-axis] = 0.0

            ## Scale or translate based on mouse button
            if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):
                if self.state['mouseMode'] == pg.ViewBox.RectMode:
                    if ev.isFinish():  ## This is the final move in the drag; change the view scale now
                        #print "finish"
                        self.rbScaleBox.hide()
                        ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                        ax = self.childGroup.mapRectFromParent(ax)
                        self.showAxRect(ax)
                        self.axHistoryPointer += 1
                        self.axHistory = self.axHistory[:self.axHistoryPointer] + [ax]
                    else:
                        ## update shape of scale box
                        self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                else:
                    tr = dif*mask
                    tr = self.mapToView(tr) - self.mapToView(Point(0,0))
                    x = tr.x() if mask[0] == 1 else None
                    y = tr.y() if mask[1] == 1 else None

                    self._resetTarget()
                    if x is not None or y is not None:
                        self.translateBy(x=x, y=y)
                    self.sigRangeChangedManually.emit(self.state['mouseEnabled'])

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

    sigImageChanged = QtCore.Signal()

    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        super(ImageDraw, self).__init__()
        #self.image=None
        #self.viewbox=viewbox
        self.levels = np.array([0,255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = 'row-major'
        self.removable = False

        self.parent = parent
        #kernel[1,1] = 1
        self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.parent.current_stroke = []
        self.parent.in_stroke = False

    def mouseClickEvent(self, ev):
        if self.parent.masksOn or self.parent.outlinesOn:
            #This is the code for starting to draw outlines.
            if  self.parent.loaded and (ev.button()==QtCore.Qt.RightButton or
                    ev.modifiers() == QtCore.Qt.ShiftModifier and not ev.double()):
                if not self.parent.in_stroke:
                    ev.accept()
                    self.create_start(ev.pos())
                    self.parent.stroke_appended = False
                    self.parent.in_stroke = True
                    self.drawAt(ev.pos(), ev)
                else:
                    ev.accept()
                    self.end_stroke()
                    self.parent.in_stroke = False
            elif not self.parent.in_stroke:
                y,x = int(ev.pos().y()), int(ev.pos().x())
                if y>=0 and y<self.parent.Ly and x>=0 and x<self.parent.Lx:
                    if (ev.button()==QtCore.Qt.LeftButton and ev.modifiers()==QtCore.Qt.ControlModifier
                            and not ev.double()):
                        # delete mask selected
                        idx = self.parent.cellpix[self.parent.currentZ][y,x]
                        if idx > 0:
                            self.parent.remove_cell(idx)
                    elif ev.button()==QtCore.Qt.LeftButton and self.parent.masksOn:
                        idx = self.parent.cellpix[self.parent.currentZ][int(ev.pos().y()), int(ev.pos().x())]
                        if idx > 0:
                            self.parent.unselect_cell()
                            self.parent.select_cell(idx)
                        else:
                            self.parent.unselect_cell()
                    else:
                        ev.ignore()
                        return


    def mouseDragEvent(self, ev):
        ev.ignore()
        return

    def hoverEvent(self, ev):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        if self.parent.in_stroke:
            if self.parent.in_stroke:
                # continue stroke if not at start
                self.drawAt(ev.pos())
                if self.is_at_start(ev.pos()):
                    self.end_stroke()
                    self.parent.in_stroke = False
        else:
            ev.acceptClicks(QtCore.Qt.RightButton)
            #ev.acceptClicks(QtCore.Qt.LeftButton)

    def create_start(self, pos):
        self.scatter = pg.ScatterPlotItem([pos.x()], [pos.y()], pxMode=False,
                                        pen=pg.mkPen(color=(255,0,0), width=self.parent.brush_size),
                                        size=max(3*2, self.parent.brush_size*1.8*2), brush=None)
        self.parent.p0.addItem(self.scatter)

    def is_at_start(self, pos):
        thresh_out = max(6, self.parent.brush_size*3)
        thresh_in = max(3, self.parent.brush_size*1.8)
        # first check if you ever left the start
        if len(self.parent.current_stroke) > 3:
            stroke = np.array(self.parent.current_stroke)
            dist = (((stroke[1:,1:] - stroke[:1,1:][np.newaxis,:,:])**2).sum(axis=-1))**0.5
            dist = dist.flatten()
            #print(dist)
            has_left = (dist > thresh_out).nonzero()[0]
            if len(has_left) > 0:
                first_left = np.sort(has_left)[0]
                has_returned = (dist[max(4,first_left+1):] < thresh_in).sum()
                if has_returned > 0:
                    return True
                else:
                    return False
            else:
                return False

    def end_stroke(self):
        self.parent.p0.removeItem(self.scatter)
        if not self.parent.stroke_appended:
            self.parent.strokes.append(self.parent.current_stroke)
            self.parent.stroke_appended = True
            self.parent.current_stroke = np.array(self.parent.current_stroke)
            ioutline = self.parent.current_stroke[:,3]==1
            self.parent.current_point_set.extend(list(self.parent.current_stroke[ioutline]))
            self.parent.current_stroke = []
            if self.parent.autosave:
                self.parent.add_set()
        if len(self.parent.current_point_set) > 0 and self.parent.autosave:
            self.parent.add_set()

    def tabletEvent(self, ev):
        pass
        #print(ev.device())
        #print(ev.pointerType())
        #print(ev.pressure())

    def drawAt(self, pos, ev=None):
        '''
        Code for drawing the outline of a mask.
        '''
        mask = self.greenmask
        set = self.parent.current_point_set
        stroke = self.parent.current_stroke
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]
        kcent = kc.copy()
        if tx[0]<=0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx    = sx
            kcent[0] = 0
        if ty[0]<=0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty    = sy
            kcent[1] = 0
        if tx[1] >= self.parent.Ly-1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.Ly - kc[0] - 1
            tx[1] = self.parent.Ly
            kcent[0] = tx[1]-tx[0]-1
        if ty[1] >= self.parent.Lx-1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.Lx - kc[1] - 1
            ty[1] = self.parent.Lx
            kcent[1] = ty[1]-ty[0]-1


        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        self.image[ts] = mask[ss]

        for ky,y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx,x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx==kcent[0], ky==kcent[1])
                stroke.append([self.parent.currentZ, x, y, iscent])
        self.updateImage()

    def setDrawKernel(self, kernel_size=3):
        bs = kernel_size
        kernel = np.ones((bs,bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [int(np.floor(kernel.shape[0]/2)),
                                 int(np.floor(kernel.shape[1]/2))]
        onmask = 255 * kernel[:,:,np.newaxis]
        offmask = np.zeros((bs,bs,1))
        opamask = 100 * kernel[:,:,np.newaxis]
        self.redmask = np.concatenate((onmask,offmask,offmask,onmask), axis=-1)
        self.greenmask = np.concatenate((onmask,offmask,onmask,opamask), axis=-1)




def run(image=None):
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QtGui.QApplication(sys.argv)

    main_window=MainW(image=image)
    main_window.show()
    ret = app.exec_()
    #sys.exit(ret)

run()
