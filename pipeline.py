import ants
import numpy as np
import h5py
import time
from zebrafish_io import lif_read_stack

def pipeline(file_path):
    stack,spacing=lif_read_stack(file_path)

pipeline('/Users/koesterlab/Documents/Maria/files/fish37_6dpf_medium.lif')
