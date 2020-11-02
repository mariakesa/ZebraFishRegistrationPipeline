from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
#import vispy.mpl_plot as plt
from PyQt5 import QtCore, QtGui, QtWidgets


import time
import numpy as np
from vispy import gloo, app
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

        cm=color.get_colormap("cool").map(self.time_s_colors[:,0])

        self.image=scene.visuals.Image(self.raw_data[0,:,:],parent=self.view.scene, cmap='grays',clim=[0,255])
        self.image.set_gl_state('translucent', depth_test=False)

        colors=vispy.color.ColorArray(cm,alpha=0.8)
        Scatter2D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        self.p1 = Scatter2D(parent=self.view.scene)
        self.p1.set_data(self.rois_plane[:,:2], face_color=colors, symbol='o', size=8,
            edge_width=0.5, edge_color='blue')


    def create_cell_image(self):
        self.cell_act=np.zeros((1024,1024))
        self.cell_act[self.rois_plane[:,0],self.rois_plane[:,1]]=1
        self.cell_act=np.rot90(self.cell_act,3)

    def load_data(self):
        filename='C:/Users/koester_lab/Documents/Maria/registered/fish2_6dpf_medium_aligned.h5'
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Loading raw data from a plane...")
            start=time.time()
            self.raw_data=f['data'][:,self.plane_ind,:,:].astype('float32')
            end=time.time()
            print('Time to load raw data file: ',end-start)
        for j in range(0,self.raw_data.shape[0]):
            self.raw_data[j,:,:] *= 400.0/(self.raw_data[j,:,:].max()+0.00001)

        self.time_s=np.load('C:/Users/koester_lab/Documents/Maria/segmented/fish2_6dpf_medium_masked_traces.npy')
        self.pos=np.load('C:/Users/koester_lab/Documents/Maria/segmented/fish2_6dpf_medium_masked_rois.npy')
        single_plane=self.pos[:,2]==self.plane_ind
        self.rois_plane=self.pos[single_plane]
        self.time_s=self.time_s[single_plane]
        self.time_s_colors=self.time_s/800







class MainWindow(QMainWindow):
    def __init__(self, canvas=None,parent=None):
        super(MainWindow, self).__init__(parent)
        self.canvas=canvas
        widget = QWidget()
        self.setCentralWidget(widget)
        self.l0 = QGridLayout()
        self.l0.addWidget(canvas.native)
        widget.setLayout(self.l0)
        self.timer_init()

    def timer_init(self):
        self.timer = vispy.app.Timer()
        self.timer.connect(self.update)
        self.timer.start(0)
        self.timer.interval=0.1

    def update(self,ev):
        cm=color.get_colormap("cool").map(canvas.time_s_colors[:,canvas.i])
        colors=vispy.color.ColorArray(cm,alpha=0.8)
        canvas.p1.set_data(canvas.rois_plane[:,:2], face_color=colors, symbol='o', size=8,
            edge_width=0.5, edge_color='blue')
        canvas.image.set_data(canvas.raw_data[canvas.i,:,:])
        print(canvas.i)
        canvas.i+=1
        if canvas.i>=canvas.raw_data.shape[0]:
            canvas.i=0



canvas = Canvas()
#view = canvas.add_view()
#image = scene.visuals.Image(im, cmap='grays', parent=view.scene)
#canvas.is_interactive(True)
vispy.use('PyQt5')
def update(ev):
    cm=color.get_colormap("cool").map(canvas.time_s_colors[:,canvas.i])
    colors=vispy.color.ColorArray(cm,alpha=0.8)
    canvas.p1.set_data(canvas.rois_plane[:,:2], face_color=colors, symbol='o', size=8,
        edge_width=0.5, edge_color='blue')
    canvas.image.set_data(canvas.raw_data[canvas.i,:,:])
    print(canvas.i)
    canvas.i+=1
    if canvas.i>=canvas.raw_data.shape[0]:
        canvas.i=0
    #global i
    #i+=1

#timer = vispy.app.Timer()
#timer.connect(update)
#timer.start(0)
#timer.interval=0.1
w = MainWindow(canvas)
w.show()
vispy.app.run()
#vispy.app.processEvents()

'''

# Create a texture
radius = 32
im1 = np.random.normal(
    0.8, 0.3, (radius * 2 + 1, radius * 2 + 1)).astype(np.float32)

# Mask it with a disk
L = np.linspace(-radius, radius, 2 * radius + 1)
(X, Y) = np.meshgrid(L, L)
im1 *= np.array((X ** 2 + Y ** 2) <= radius * radius, dtype='float32')

# Set number of particles, you should be able to scale this to 100000
N = 18795

# Create vertex data container
data = np.zeros(N, [('a_position', np.float32, 3),
                    ('a_color', np.float32, 4),
                    ('a_lifetime',np.float32)])


VERT_SHADER = """
uniform float u_time;
attribute vec3 a_position;
attribute vec4 a_color;
attribute float a_lifetime;
varying float color_;
varying float v_lifetime;

void main () {
    gl_Position.xyz = a_position;

    color_=a_color.a;

    //gl_PointSize = 5.0;

    v_lifetime = 1.0 - (u_time / a_lifetime);
    v_lifetime = clamp(v_lifetime, 0.0, 1.0);

    gl_PointSize = (v_lifetime * v_lifetime) * 20.0;

}
"""

from vispy.app import use_app
use_app('PyQt5')
# Deliberately add precision qualifiers to test automatic GLSL code conversion
FRAG_SHADER = """
#version 120
uniform vec4 u_color;
precision highp float;
uniform sampler2D texture1;
varying float color_;
uniform highp sampler2D s_texture;
void main()
{
    highp vec4 texColor;
    texColor = texture2D(s_texture, gl_PointCoord);
    gl_FragColor = vec4(u_color) * texColor;
    gl_FragColor.a = color_;

}
"""
global i
i=0

class Canvas(app.Canvas):

    def __init__(self,ens_n):
        app.Canvas.__init__(self,keys='interactive', size=(800, 600))

        # Create program
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._program.bind(gloo.VertexBuffer(data))
        self._program['s_texture'] = gloo.Texture2D(im1)
        self.transp=np.load('my_spks_.npy')
        print('transp',self.transp.shape)

        self.pos=((np.load('pos.npy')[:])*2)-1
        self.i=0

        self.ens_n=ens_n
        self.ensemble=np.load('U.npy')[:,ens_n]
        print('U shp',self.ensemble.shape)
        # Create first explosion
        self._new_explosion()

        # Enable blending
        gloo.set_state(blend=True, clear_color='black',
                       blend_func=('src_alpha', 'one'))

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])

        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.i=0
        global i
        i=0
        #self.show()

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):

        # Clear
        gloo.clear()

        # Draw
        self._program['u_time'] = time.time() - self._starttime

        # New explosion?
        if time.time() - self._starttime >1:
            self._new_explosion()

        self._program.draw('points')

    def _new_explosion(self):
        global i
        print(i)
        i+=1
        #self.i+=1

        #self._program['u_centerPosition'] = centerpos

        # New color, scale alpha with N
        a_transp=self.transp[i,:]

        color=np.ones((18795,4))
        print('ens',self.ensemble.shape)

        color[:,3]=a_transp*self.ensemble

        alpha = 1.0 / N ** 0.08
        color_un = np.random.uniform(0.1, 0.9, (3,))

        self._program['u_color'] = tuple(color_un) + (alpha,)
        #self._program['color'] = color.astype('float32')

        # bind the VBO to the GL context
        #self._program.bind(self.data_vbo)
        data['a_color'] = color

        print(color)

        data['a_lifetime'] = np.random.normal(2.0, 0.5, (N,))

        data['a_position'] = self.pos

        self._program.bind(gloo.VertexBuffer(data))

        # Set time to zero
        self._starttime = time.time()

    def set_data(self,ens_n):
        if ens_n!=-1:
            self.ensemble=np.load('U.npy')[:,ens_n]
        else:
            self.ensemble=np.ones(18795,)


from PyQt5.QtWidgets import *
import vispy.app
import sys

class MainWindow(QMainWindow):
    def __init__(self, canvas=None,parent=None):
        super(MainWindow, self).__init__(parent)
        self.canvas=canvas
        widget = QWidget()
        self.setCentralWidget(widget)
        self.l0 = QGridLayout()
        self.l0.addWidget(canvas.native)
        widget.setLayout(self.l0)

        self.n_ens=QLineEdit()
        self.n_ens.setText("25")
        self.n_ens.setFixedWidth(35)
        self.l0.addWidget(self.n_ens, 0, 4.5, 1, 2)
        self.n_ens.returnPressed.connect(lambda: self.on_set_n_ens())
        self.n_components=int(self.n_ens.text())

        #vispy.app.KeyEvent('returnPressed')
        #self.canvas.connect(self.on_set_n_ens())
        #self.n_ens.valueChanged.connect(self.on_set_n_ens)

        self.update_view()


    def on_set_n_ens(self):
        self.n_components=int(self.n_ens.text())
        print('n_components',self.n_components)
        #self.show()
        self.update_view()

    def update_view(self):
        global i
        print(i)
        i=0
        self._starttime = time.time()
        self.canvas.set_data(self.n_components)

ens_n=0
canvas = Canvas(ens_n)
#canvas.is_interactive(True)
vispy.use('PyQt5')
w = MainWindow(canvas)
w.show()
vispy.app.run()
vispy.app.processEvents()

'''
