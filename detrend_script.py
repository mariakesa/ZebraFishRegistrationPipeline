import h5py
import numpy as np
from scipy.signal import detrend
import time

file_path='//ZMN-HIVE/User-Data/Maria/masked/fish20_6dpf_medium_masked.h5'
save_path='//ZMN-HIVE/User-Data/Maria/masked/fish20_6dpf_medium_masked_detrended.h5'

def detrend_a_plane(file_path,plane_ind):
    with h5py.File(file_path, "r") as f:
        print("Loading raw data from plane: "+str(file_path))
        start=time.time()
        dat=f['data'][:,plane_ind,:,:].astype('float32')
        end=time.time()
        print('Time to load raw data file: ',end-start)
    dat=dat.reshape(-1,1024*1024)
    #Add the mean to have on the same scale as the original data for cell segmentation purposes
    #detr=detrend(dat,axis=0)+np.mean(np.mean(dat,axis=1))
    detr=detrend(dat)
    return detr.reshape(-1,1024,1024)

def detrend_file(file_path,save_path):
    with h5py.File(file_path, "r") as f:
        print("Loading raw data from plane: "+str(file_path))
        start=time.time()
        dat=f['data'][:,0,0,0].astype('float32')
        end=time.time()
        print('Time to load raw data file: ',end-start)
    n_timepoints=np.array(dat).shape[0]
    detr_container = np.zeros((n_timepoints,21,1024,1024),dtype='float32')

    for z in range(0,21):
        detr_container[:,z,:,:]= detrend_a_plane(file_path,z)


    detrended = h5py.File(save_filename, 'w')
    detrended.create_dataset('data',data=detr_container)
    detrended.close()

detrend_file(file_path,save_path)
