import argparse
import deepdish as dd
import numpy as np
import os

"""
Call this with:
bokeh serve --show roi_explorer.py --args yourh5file.h5
"""

parser = argparse.ArgumentParser(description="Display h5 fish file")
parser.add_argument("file", help="The path to the hdf5 file")
args = parser.parse_args()

aligned_fn = args.file

"""
base2 = base.replace('registered', 'segmented')
rois_fn = base2 + '_rois.npy'
traces_fn = base2 + '_traces.npy'

rois = np.load(rois_fn)
traces = np.load(traces_fn)
"""

def load(t, z):
    return dd.io.load(aligned_fn, sel=(slice(t, t + 1), slice(z, z + 1))).squeeze()

from bokeh.layouts import column, row, layout
from bokeh.models import ColumnDataSource, Range1d, WheelZoomTool
from bokeh.models.widgets import Slider
from bokeh.plotting import curdoc, figure

view_figure = figure(plot_width=950, plot_height=950, x_range=Range1d(0, 1024, bounds='auto'),
           y_range=Range1d(1024, 0, bounds='auto'))
img_src = ColumnDataSource({})
view_figure.toolbar.active_scroll = view_figure.select_one(WheelZoomTool)
view_figure.toolbar_location = None

"""
rois_data = []
for z in range(21):
    x = [r[0] for r in rois if r[2] == z]
    y = [r[1] for r in rois if r[2] == z]
    rois_data.append((x, y))
"""

img = view_figure.image('value', x=0, y=1024, dw=1024, dh=1024, source=img_src, palette='Greys256')

"""
rois_src = ColumnDataSource(dict(x=[], y=[]))
rois_glyph = view_figure.circle(x='x', y='y', size=2, color='red', source=rois_src)

hist_figure = figure(plot_width=800, plot_height=250)
hist_src = ColumnDataSource(dict(x=[], top=[]))
hist = hist_figure.vbar(x='x', top='top', source=hist_src, width=1)
"""

minv_slider = Slider(start=0, end=1000, value=0, step=1, title='Min', width=900)
maxv_slider = Slider(start=0, end=1000, value=1000, step=1, title='Max', width=900)

global z,t
z = 10
t = 0

def update_rois():
    """
    global z
    x, y = rois_data[z]
    rois_src.data = dict(x=x, y=y)
    """
    pass

def update_img(new_hist=False):
    global z, t
    print('z',z)
    print('t',t)
    new_data = load(t, z)

    global minv_slider, maxv_slider
    minv = minv_slider.value
    maxv = maxv_slider.value

    displayed = display_image(new_data, minv, maxv)
    img_src.data = {'value': [displayed]}

    """
    if new_hist:
        hist_data = np.bincount(new_data.flatten())
        hist_src.data = dict(x=np.arange(np.alen(hist_data)), top=hist_data)
    """

def display_image(img, min_v, max_v):
    f = 255.0/(max_v-min_v)
    return np.flip((np.array(np.maximum(np.minimum(img, max_v), min_v)-min_v)*f).astype(np.float32), axis=0)

def t_select_handler(attr, old, new):
    global t
    t = new
    update_img(True)

def z_select_handler(attr, old,new):
    global z
    z = new
    update_img(True)
    #update_rois()

def handler(attr, old, new):
    update_img()

update_img(True)
#update_rois()

t_select = Slider(start=0, end=1799, value=0, step=1, title='Time', width=950)
t_select.on_change('value', t_select_handler)

z_select = Slider(start=0, end=20, value=0, step=1, title='Z', width=950)
z_select.on_change('value', z_select_handler)

minv_slider.on_change('value', handler)
maxv_slider.on_change('value', handler)

l = layout([
    [view_figure],
    [t_select],
    [minv_slider],
    [maxv_slider],
    [z_select]
])

curdoc().add_root(l)
curdoc().title = os.path.basename(args.file)
