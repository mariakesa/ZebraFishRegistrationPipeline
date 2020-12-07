'''
Code inspired and adapted from Carsen Stringer CellPose https://github.com/MouseLand/cellpose/blob/master/cellpose/gui.py

https://stackoverflow.com/questions/50847827/how-can-i-select-the-pixels-that-fall-within-a-contour-in-an-image-represented-b
https://stackoverflow.com/questions/31542843/inpolygon-for-python-examples-of-matplotlib-path-path-contains-points-method
'''

from PyQt5 import QtGui, QtCore, Qt, QtWidgets
import pyqtgraph as pg
import numpy as np
import sys
import h5py
import time
from matplotlib import image
import matplotlib.path
from pyqtgraph import Point
from collections.abc import Callable
import copy
from matplotlib import cm

class MainW(QtGui.QMainWindow):
    def __init__(self,filename,save_masked_filename,save_mask_filename,reg_t_ind):
        super(MainW, self).__init__()

        pg.setConfigOptions(imageAxisOrder="col-major")
        self.setGeometry(50, 50, 1200, 1000)

        self.setStyleSheet("QMainWindow {background: 'black';}")

        self.filename=filename
        self.save_masked_filename= save_masked_filename
        self.save_mask_filename=save_mask_filename
        self.reg_t_ind=reg_t_ind

        self.load_image()

        self.shp=self.data.shape
        print('data shape: ', self.shp)
        #Data and output arrays
        self.mask_arr=np.zeros((self.shp[0],self.shp[1],self.shp[2])).astype('float64')

        colormap = cm.get_cmap("bwr")
        colormap._init()
        self.lut = 255*(colormap._lut).view(np.ndarray)[:255]

        #Plane ind
        self.plane_ind=0

        self.cwidget = QtGui.QWidget(self)
        self.l0 = QtGui.QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.setCentralWidget(self.cwidget)

        self.win = pg.GraphicsLayoutWidget()

        self.l0.addWidget(self.win,0,0,40,10)

        self.make_viewbox()

        self.make_buttons()

        self.make_plane_input()

        self.make_plane_text_box()

        self.set_image()

        self.win.show()

    def load_image(self):
        self.reg_t_ind=0
        with h5py.File(self.filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            start=time.time()
            data=f['data'][self.reg_t_ind,:,:,:]
            end=time.time()
            print('Time to load file: ',end-start)
        self.data=np.array(data).astype('float64')
        for j in range(0,21):
            self.data[j,:,:] *= 1000.0/self.data[j,:,:].max()
        self.data_masked=copy.deepcopy(self.data)

    def set_image(self):
        self.img.setImage(self.data_masked[0,:,:], autoLevels=False, levels=[0,255])
        self.img.setLookupTable(self.lut)
        self.show()

    def make_viewbox(self):
        self.p0 = pg.ViewBox(invertY=True)
        self.brush_size=3
        self.win.addItem(self.p0, 0, 0)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.p0.setAspectLocked(True)
        self.img = ImageDraw(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        kern = 255*np.ones((3,3))
        self.img.setDrawKernel(kern, mask=kern, center=(1,1), mode='add')
        self.p0.addItem(self.img)

    def make_buttons(self):
        self.segment_button=QtGui.QPushButton('Segment')
        self.l0.addWidget(self.segment_button, 52, 0,2,2)
        self.l0.setColumnStretch(2,1)
        self.segment_button.clicked.connect(lambda: self.segment())
        self.clear_button=QtGui.QPushButton('Clear')
        self.l0.addWidget(self.clear_button, 54, 0,2,2)
        self.clear_button.clicked.connect(lambda: self.clear_img())
        self.mask_all_button=QtGui.QPushButton('Mask all')
        self.l0.addWidget(self.mask_all_button, 56, 0,2,2)
        self.mask_all_button.clicked.connect(lambda: self.mask_all_data())

    def make_plane_input(self):
        self.plane_input=QtGui.QLineEdit(self)
        self.plane_input.setText("0")
        self.plane_input.setFixedWidth(35)
        self.plane_input.returnPressed.connect(lambda: self.set_plane_ind_image())
        self.l0.addWidget(self.plane_input, 50, 0,1,1)
        self.l0.setVerticalSpacing(1)

    def set_plane_ind_image(self):
        self.plane_ind=int(self.plane_input.text())
        self.img.setImage(self.data_masked[self.plane_ind,:,:], autoLevels=False, lut=self.lut,levels=[0,255])
        self.show()


    def segment(self):

        print('Segmenting plane: ', self.plane_ind)

        x=np.arange(0,self.shp[1])
        y=np.arange(0,self.shp[2])
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))

        path = matplotlib.path.Path(self.img.pt_lst)
        mask = path.contains_points(points)
        mask=mask.reshape(self.shp[1],self.shp[2]).astype('float64')

        self.data_masked[self.plane_ind,:,:][mask==0]=0

        self.mask_arr[self.plane_ind]=mask

        self.img.setImage(self.data_masked[self.plane_ind,:,:],autoLevels=False, lut=self.lut,levels=[0,255])

        self.update_text_box()

        self.img.pt_lst=[]

    def make_plane_text_box(self):
        self.plane_text = QtGui.QTextEdit()
        self.cursor = self.plane_text.textCursor()
        self.plane_text.ensureCursorVisible()
        self.l0.addWidget(self.plane_text, 58,0,5,1)

    def update_text_box(self):
        text=self.plane_text.toPlainText()+'\n'+'Masked plane nr: '+str(self.plane_ind)
        nrs=[int(s) for s in text.split() if s.isdigit()]
        sortd=sorted(nrs)
        self.plane_text.clear()
        for nr in sortd:
            self.cursor.movePosition(self.cursor.End)
            self.cursor.insertText('\n'+'Masked plane nr: '+str(nr))
            self.plane_text.ensureCursorVisible()

    def delete_from_text_box(self):
        text=self.plane_text.toPlainText()+'\n'+'Masked plane nr: '+str(self.plane_ind)
        nrs=[int(s) for s in text.split() if s.isdigit() and s!=str(self.plane_ind)]
        sortd=sorted(nrs)
        self.plane_text.clear()
        for nr in sortd:
            self.cursor.movePosition(self.cursor.End)
            self.cursor.insertText('\n'+'Masked plane nr: '+str(nr))
            self.plane_text.ensureCursorVisible()

    def mask_all_data(self):
        with h5py.File(self.filename, "r+") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            start=time.time()
            data=f['data'][()]
            end=time.time()
            print('Time to load file: ',end-start)
        shp=data.shape
        start=time.time()
        for plane_ind in range(shp[1]):
            for time_point in range(shp[0]):
                data[time_point,plane_ind,:,:][self.mask_arr[plane_ind,:,:]==0]=0
        end=time.time()
        print('Time for masking: ', end-start)
        start=time.time()
        for_segmentation = h5py.File(self.save_masked_filename, 'w')
        for_segmentation.create_dataset('data',data=data)
        for_segmentation.close()

        np.save(self.save_mask_filename,self.mask_arr)

        end=time.time()
        print('Time for saving: ', end-start)

        sys.exit()

    def clear_img(self):
        self.data_masked[self.plane_ind,:,:]=self.data[self.plane_ind,:,:].copy()
        self.img.pt_lst=[]
        self.img.setImage(self.data_masked[self.plane_ind,:,:],autoLevels=False,levels=[0,255])

        self.delete_from_text_box()

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

    def setLookupTable(self, lut, update=True):
        """
        Set the lookup table (numpy array) to use for this image. (see
        :func:`makeARGB <pyqtgraph.makeARGB>` for more information on how this is used).
        Optionally, lut can be a callable that accepts the current image as an
        argument and returns the lookup table to use.

        Ordinarily, this table is supplied by a :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>`
        or :class:`GradientEditorItem <pyqtgraph.GradientEditorItem>`.
        """
        if lut is not self.lut:
            self.lut = lut
            self._effectiveLut = None
            if update:
                self.updateImage()

def run_mask_gui(filename,saved_masked_filename,save_mask_filename,reg_t_ind):
    app = QtGui.QApplication(sys.argv)
    main_window=MainW(filename,saved_masked_filename,save_mask_filename,reg_t_ind)
    main_window.show()
    ret = app.exec_()
    sys.exit(ret)
