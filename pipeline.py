import ants
import numpy as np
import h5py
import time
from zebrafish_io import lif_read_stack, save

def pipeline(file_path,fixed_index,moving_indices,rigid_out_file,atlas_file):
    #stack,spacing=lif_read_stack(file_path)
    #rigid=rigid_registration(stack, fixed_index, moving_indices,rigid_out_file,spacing)
    get_diffeomorphic_transform(atlas_file,rigid_out_file)

def rigid_registration(stack, fixed_index, moving_indices,out_file,spacing):
    fixed=stack[fixed_index,:,:,:].astype(np.float32)
    moving=stack[moving_indices,:,:,:].astype(np.float32)
    shape=(moving.shape[0]+1,moving.shape[1],moving.shape[2],moving.shape[3])
    rigid = np.empty(shape, dtype=np.float32)
    rigid[0,:,:,:]=fixed
    fixed=ants.from_numpy(fixed)

    start=time.time()
    for timepoint in range(moving.shape[0]-1):
        print('Iteration: ', timepoint)
        moving_im=moving[timepoint,:,:,:]
        moving_im=ants.from_numpy(moving_im)
        rigid_transform = ants.registration(fixed=fixed, moving=moving_im,
        type_of_transform = 'Rigid')
        transformed_image=rigid_transform['warpedmovout']
        im=transformed_image.numpy()
        rigid[timepoint+1,:,:,:]=im

    end=time.time()
    print('time: ', end-start)

    save(out_file, rigid, spacing)

    return rigid

def get_diffeomorphic_transform(atlas_file,rigid_out_file):
    stack = h5py.File(rigid_out_file, 'r')
    one_image = np.array(stack['ITKImage']['0']['VoxelData'])[0,:,:,:]
    atlas=np.array(h5py.File(atlas_file, 'r')['warped_image'])
    #data = hf.get('warped_image')[()]

def morph_timestack(atlas_file,rigid_out_file,saved_transform):
    pass

file_path='/Users/koesterlab/Documents/Maria/files/fish37_6dpf_medium.lif'
fixed_index=0
moving_indices=range(1,10)
#176
#195
rigid_out_file='/Users/koesterlab/Documents/Maria/files/fish37_6dpf_medium_rigid_0_10.h5'
atlas_file='/Users/koesterlab/Documents/Maria/files/test_16_atlas_highres_z.h5'
pipeline(file_path,fixed_index,moving_indices,rigid_out_file,atlas_file)
