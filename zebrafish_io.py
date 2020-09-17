import bioformats
import deepdish as dd
import h5py
import javabridge
import numpy as np
import os.path
from pyprind import prog_percent
import SimpleITK as sitk
import tables
import time
from xml.etree import ElementTree as ETree

def lif_get_metas(fn):
    md = bioformats.get_omexml_metadata(fn)  # Load meta data
    mdroot = ETree.fromstring(md)  # Parse XML
    #    meta = mdroot[1][3].attrib # Get relevant meta data
    metas = list(map(lambda e: e.attrib, mdroot.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')))

    return metas

def lif_find_timeseries(fn):
    metas = lif_get_metas(fn)

    meta = None
    img_i = 0
    for i, m in enumerate(metas):
        if int(m['SizeT']) > 1:
            meta = m
            img_i = i

    if not meta:
        raise ValueError('lif does not contain an image with sizeT > 1')

    return img_i

def start_jvm():
    javabridge.start_vm(class_path=bioformats.JARS)

    log_level = 'ERROR'
    # reduce log level
    """
    rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
    rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                        "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
    logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", log_level, "Lch/qos/logback/classic/Level;")
    javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)
    """


def lif_open(fn):
    start_jvm()
    ir = bioformats.ImageReader(fn)

    return ir

def lif_read_stack(fn):
    ir = lif_open(fn)
    img_i = lif_find_timeseries(fn)
    shape, spacing = get_shape(fn, img_i)

    stack = np.empty(shape, dtype=np.uint16)

    # Load the whole stack...
    for t in prog_percent(range(stack.shape[0])):
        for z in range(stack.shape[1]):
            stack[t, z] = ir.read(t=t, z=z, c=0, series=img_i, rescale=False)

    return stack, spacing

def get_shape(fn, index=0):
    """

    :param fn: image file
    :return: shape of that file
    """
    in_ext = os.path.splitext(fn)[1]

    if in_ext == '.h5':
        """
        f = tables.open_file(fn)
        return f.get_node('/stack').shape
        """
        img = load(fn)
        return img.shape
    elif in_ext == '.nrrd':
        img = load(fn)
        return img.shape
    elif in_ext == '.lif':
        metas = lif_get_metas(fn)
        meta = metas[index]

        shape = (
            int(meta['SizeT']),
            int(meta['SizeZ']),
            int(meta['SizeY']),
            int(meta['SizeX']),
        )
        order = meta['DimensionOrder']
        spacing = tuple([float(meta['PhysicalSize%s' % c]) for c in 'XYZ'])

        return shape, spacing

    else:
        raise UnsupportedFormatException('Input format "' + in_ext + '" is not supported.')


def __sitkread(filename):
     img = sitk.ReadImage(filename)
     spacing = img.GetSpacing()
     return sitk.GetArrayFromImage(img), spacing


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
