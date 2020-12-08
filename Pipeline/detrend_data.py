import h5py
import numpy as np
from scipy.signal import detrend
import time

def check_dim(file_path):
    with h5py.File(file_path, "r") as f:
        print("Loading raw data from plane: "+str(file_path))
        start=time.time()
        dat=f['data'][0,:,:,:].astype('float32')
        end=time.time()
        print('Time to load raw data file: ',end-start)
    return dat.shape[0], dat.shape[1], dat.shape[2]

def detrend_a_plane(file_path,plane_ind):
    with h5py.File(file_path, "r") as f:
        print("Loading raw data from plane: "+str(file_path))
        start=time.time()
        dat=f['data'][:,plane_ind,:,:].astype('float32')
        end=time.time()
        print('Time to load raw data file: ',end-start)
    x_dim=dat.shape[1]
    y_dim=dat.shape[2]
    dat=dat.reshape(-1,x_dim*y_dim)
    #Add the mean to have on the same scale as the original data for cell segmentation purposes
    #detr=detrend(dat,axis=0)+np.mean(np.mean(dat,axis=1))
    detr=detrend(dat,axis=0)
    return detr.reshape(-1,x_dim,y_dim)

def detrend_file(file_path,save_path):
    n_planes,x_dim,y_dim=check_dim(file_path)
    with h5py.File(file_path, "r") as f:
        print("Loading raw data from plane: "+str(file_path))
        start=time.time()
        dat=f['data'][:,0,0,0].astype('float32')
        end=time.time()
        print('Time to load raw data file: ',end-start)
    n_timepoints=np.array(dat).shape[0]
    detr_container = np.zeros((n_timepoints,n_planes,x_dim,y_dim),dtype='float32')

    for z in range(0,n_planes):
        print('Working on plane: ', z)
        detr_container[:,z,:,:]= detrend_a_plane(file_path,z)


    detrended = h5py.File(save_path, 'w')
    detrended.create_dataset('data',data=detr_container)
    detrended.close()
