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
