import numpy as np
import os
from skimage.draw import circle
from skimage.io import imread
from skimage.feature import match_template, peak_local_max, blob_log
from sklearn.preprocessing import normalize
import SimpleITK as sitk
import deepdish as dd
import time

STD_DEV_SUFFIX = '_std_dev.h5'
ROIS_SUFFIX = '_rois.npy'
TRACES_SUFFIX = '_traces.npy'

def __sitkwrite(filename, data, spacing):
    img = sitk.GetImageFromArray(data)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, filename)

def save(fn, data, spacing):
    out_ext = os.path.splitext(fn)[1]

    if out_ext == '.nrrd':
        __sitkwrite(fn, data, spacing)
    elif out_ext == '.h5':
        """
        with tables.open_file(fn, mode='w') as f:
            f.create_array('/', 'stack', data.astype(np.float32))
            f.close()
        """
        __sitkwrite(fn, data, spacing)
    else:
        raise UnsupportedFormatException('Output format "' + out_ext + '" is not supported.')

def std(stack, valid_frames=None):
    anatomy_std = []

    if not valid_frames:
        valid_frames = np.arange(np.alen(stack))

    for plane in range(stack.shape[1]):
        #    anatomy_std.append(np.std(trace[displacement[plane]<30, plane], axis=0))
        anatomy_std.append(np.std(stack[valid_frames, plane, ...], axis=0))

    anatomy_std = np.array(anatomy_std, dtype=np.float32)
    return anatomy_std

def find_rois_template(nda, template, peakMinDistance=2, peakRelThreshold=.2, minIntensity=20):
    '''
    ROI detection adapted from andreas.
    :param nda:
    :param template:
    :param peakMinDistance:
    :param peakRelThreshold:
    :param minIntensity:
    :return:
    '''
    rois = []
    for z, img in enumerate(nda):
        m = match_template(img, template, pad_input=True)
        plm = peak_local_max(m, min_distance=peakMinDistance, threshold_rel=peakRelThreshold)
        rois.extend(np.asarray([(y, x, z, 0) for x, y in plm if (img[x, y] > minIntensity)]))
    return np.array(rois)

def get_traces(stack, rois, use_radius=None):
    traces = []
    for roi in rois:
        x, y, z, r = list(map(int, roi))
        if use_radius != None:
            r = use_radius
        stencil = circle(y, x, r, stack[0][z].shape)
        trace = np.array(stack[:, z, stencil[0], stencil[1]].mean(1))
        traces.append(trace)
    return np.asarray(traces)

def load_template():
    template_fn = os.path.dirname(__file__) + '/template/cell.tif'
    if not os.path.exists(template_fn):
        raise Exception('Error: Template file not found.')

    template = imread(template_fn)
    return template

def segmentation(data_path,save_path_std,save_path_roi,save_path_traces):
    print('Loading file...')
    start=time.time()
    stack = dd.io.load(data_path)['data']
    end=time.time()
    print('Done loading data in ',end-start,' seconds!')
    spacing_jakob=(0.7188675, 0.7188675, 10)
    template=load_template()
    std_ = std(stack)
    print('Saving std...')
    save(save_path_std, std_, spacing=spacing_jakob)
    print('Finding rois...')
    rois = find_rois_template(std_, template=template)
    print('Saving rois...')
    np.save(save_path_roi, rois)
    print('Getting traces...')
    traces = get_traces(stack, rois, use_radius=5)
    print('Saving traces...')
    np.save(save_path_traces, traces)
    print('Done with saving the files!')
