import h5py
import numpy as np
import time

file_path='//ZMN-HIVE/User-Data/Maria/masked/fish20_6dpf_medium_masked.h5'
save_path='//ZMN-HIVE/User-Data/Maria/masked/fish20_6dpf_medium_masked_cut.h5'

def cut(file_path,save_path):
    with h5py.File(file_path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        start=time.time()
        data=f['data'][()]
        end=time.time()
        print('Time to load file: ',end-start)
    cut_ind=np.argmin(np.array(data[:,10,500,500]),axis=0)
    print(cut_ind)
    cut_arr=data[cut_ind:,:,:,:]
    cut_file = h5py.File(save_path, 'w')
    cut_file.create_dataset('data',data=cut_arr)
    cut_file.close()

cut(file_path,save_path)
