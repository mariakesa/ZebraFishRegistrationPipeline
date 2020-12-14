import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import time

def detrending_verify(filename,save_folder_detrending,save_folder_masked,plane_ind=10):
    f_str=os.path.split(filename)[-1]
    mask_f_str=f_str.replace('aligned.h5','masked.h5')
    detr_f_str=f_str.replace('aligned.h5','detrended.h5')
    masked_file=os.path.join(os.path.normpath(save_folder_masked),mask_f_str)
    detr_file=os.path.join(os.path.normpath(save_folder_detrending),detr_f_str)
    with h5py.File(detr_file, "r") as f:
            print("Loading detrended data from plane: "+str(plane_ind))
            start=time.time()
            detrended=f['data'][:,plane_ind,:,:].astype('float32')
            end=time.time()
            print('Time to load detrended plane data file: ',end-start)
    with h5py.File(masked_file, "r") as f:
            print("Loading masked data from plane: "+str(plane_ind))
            start=time.time()
            masked=f['data'][:,plane_ind,:,:].astype('float32')
            end=time.time()
            print('Time to load masked plane data file: ',end-start)
    detrended=detrended.reshape(-1,1024*1024)
    masked=masked.reshape(-1,1024*1024)
    plt.plot(np.mean(detrended,axis=1),
    label='Detrended')
    plt.plot(np.mean(masked,axis=1),
    label='Masked')
    plt.legend()
    plt.title('Detrending comparison (average image intensity in time) for plane'+str(plane_ind))
    plt.show()
