import ants
import numpy as np
import h5py
import time
from zebrafish_io import lif_read_stack, save
import multiprocessing as mp

def pipeline(file_path,fixed_index,moving_indices,rigid_out_file,atlas_file,transform_out_file,out_diff_file):
    stack,spacing=lif_read_stack(file_path)
    #rigid=rigid_registration(stack, fixed_index, moving_indices,rigid_out_file,spacing)
    #get_diffeomorphic_transform(atlas_file,rigid_out_file,transform_out_file)
    get_diffeomorphic_transform(stack, atlas_file,transform_out_file)
    morph_timestack(atlas_file,rigid_out_file,saved_transform,out_diff_file,spacing)


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


def get_diffeomorphic_transform(stack,atlas_file,transform_out_file):
    stack = h5py.File(rigid_out_file, 'r')
    moving = np.array(stack['ITKImage']['0']['VoxelData'])[0,:,:,:]
    moving=ants.from_numpy(moving)
    atlas=np.array(h5py.File(atlas_file, 'r')['warped_image'])
    atlas=ants.from_numpy(atlas)
    start=time.time()
    diffeomorphic_transform = ants.registration(fixed=atlas , moving=moving ,
                                type_of_transform = 'SyN', syn_metric='CC', grad_step=0.25,
                                flow_sigma=6,total_sigma=0.5, reg_iterations=[200,200,200,200,10],syn_sampling=2)
    end=time.time()
    print('time: ', end-start)
    transform=diffeomorphic_transform['fwdtransforms']
    ants.write_transform(transform, transform_out_file)
    #data = hf.get('warped_image')[()]

def morph_timestack(atlas_file,rigid_out_file,saved_transform,out_diff_file,spacing):
    print('Morphing')
    stack = h5py.File(rigid_out_file, 'r')
    atlas=np.array(h5py.File(atlas_file, 'r')['warped_image'])
    atlas=ants.from_numpy(atlas)
    fixed=atlas
    moving = np.array(stack['ITKImage']['0']['VoxelData'])[0:10,:,:,:]
    transformlist=['/Users/koesterlab/Documents/Maria/files/transform.mat']
    out_diff=apply_transforms(fixed, moving, transformlist,
                     interpolator='welchWindowedSinc')
    out_diff=out_diff.numpy()
    save(out_diff_file, out_diff, spacing)

file_path='/Users/koesterlab/Documents/Maria/files/fish37_6dpf_medium.lif'
fixed_index=176
moving_indices=range(177,206)
#176
#195
#rigid_out_file='/Users/koesterlab/Documents/Maria/files/fish37_6dpf_medium_rigid_176_206.h5'
rigid_out_file='/Users/koesterlab/Documents/Maria/files/fish37_6dpf_medium_rigid_0_10.h5'
atlas_file='/Users/koesterlab/Documents/Maria/files/test_16_atlas_highres_z.h5'
transform_out_file='/Users/koesterlab/Documents/Maria/files/transform.mat'
out_diff_file='/Users/koesterlab/Documents/Maria/files/fish37_6dpf_medium_diff_0_10.h5'
pipeline(file_path,fixed_index,moving_indices,rigid_out_file,atlas_file,transform_out_file,out_diff_file)
