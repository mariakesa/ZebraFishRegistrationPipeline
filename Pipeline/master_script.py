from mask_gui import run_mask_gui
import os
import sys
from multiprocessing import Process, freeze_support
sys.path.insert(0, 'C:/Users/koester_lab/Documents/Maria/ZebraFishRegistrationPipeline/Visualization')
from movie_gui import run_movie_gui
from detrend_data import detrend_file
from segmentation import segmentation
from rastermap_comp import rastermap_comp
from dff import compute_dff
from rastermap_viz import run_rm_viz
from viz_gui import run_complete_viz
from transparent_cells import run_transparent_cells
from os.path import isfile, join
from os import listdir

def masking(filename,reg_t_ind,save_folder_mask,save_folder_masked):
    f_str=os.path.split(filename)[-1]
    #Masking
    filename=os.path.normpath(filename)
    print(filename)
    mask_f_str=f_str.replace('aligned.h5','mask.npy')
    masked_f_str=f_str.replace('aligned.h5','masked.h5')
    save_mask_filename=os.path.join(os.path.normpath(save_folder_mask),mask_f_str)
    save_masked_filename=os.path.join(os.path.normpath(save_folder_masked),masked_f_str)
    save_mask_in_folder= [f for f in listdir(os.path.normpath(save_folder_mask)) if isfile(save_mask_filename)]
    save_masked_in_folder= [f for f in listdir(os.path.normpath(save_folder_masked)) if isfile(save_masked_filename)]
    if len(save_mask_in_folder)==0 or len(save_masked_in_folder)==0:
        Process(target = run_mask_gui, args=(filename,save_masked_filename,save_mask_filename,reg_t_ind)).start()
        Process(target = run_movie_gui, args=(filename,)).start()
    else:
        print('Masked or mask file already exists!')

def detrending(filename, save_folder_mask,save_folder_detrending):
    f_str=os.path.split(filename)[-1]
    mask_f_str=f_str.replace('aligned.h5','masked.h5')
    detr_f_str=f_str.replace('aligned.h5','detrended.h5')
    masked_file=os.path.join(os.path.normpath(save_folder_mask),mask_f_str)
    save_detr_filename=os.path.join(os.path.normpath(save_folder_detrending),detr_f_str)
    detrend_file(masked_file,save_detr_filename)

def cell_extraction(filename, dat_to_seg_folder, save_folder_segmentation):
    f_str=os.path.split(filename)[-1]
    folder_str=os.path.split(dat_to_seg_folder)[-1]
    if folder_str=='detrended':
        dat_f_str=f_str.replace('aligned.h5','detrended.h5')
    elif folder_str=='masked':
        dat_f_str=f_str.replace('aligned.h5','masked.h5')
    segm_f_str=f_str.replace('aligned.h5','detrended.h5')

def segment(filename,save_folder_detrending,save_folder_segmentation):
    f_str=os.path.split(filename)[-1]
    detr_f_str=f_str.replace('aligned.h5','detrended.h5')
    save_path=os.path.join(os.path.normpath(save_folder_segmentation),f_str)
    save_path_std=save_path.replace('aligned.h5','std_dev.h5')
    save_path_roi=save_path.replace('aligned.h5','rois.npy')
    save_path_traces=save_path.replace('aligned.h5','traces.npy')
    data_path=os.path.join(os.path.normpath(save_folder_detrending),detr_f_str)
    #data_path
    segmentation(data_path,save_path_std,save_path_roi,save_path_traces)

def dff(filename,segmentation_folder,save_folder_dff):
    f_str=os.path.split(filename)[-1]
    dff_f_str=f_str.replace('aligned.h5','dff.npy')
    traces_f_str=f_str.replace('aligned.h5','traces.npy')
    save_dff_path=os.path.join(os.path.normpath(save_folder_dff),dff_f_str)
    traces_path=os.path.join(os.path.normpath(segmentation_folder),traces_f_str)
    compute_dff(traces_path,save_dff_path)

def rastermap_save(filename, dff_folder, save_folder_rastermap):
    f_str=os.path.split(filename)[-1]
    dff_f_str=f_str.replace('aligned.h5','dff.npy')
    rastermap_f_str=f_str.replace('aligned.h5','rm.npy')
    dff_path=os.path.join(os.path.normpath(dff_folder),dff_f_str)
    rastermap_save_path=os.path.join(os.path.normpath(save_folder_rastermap),rastermap_f_str)
    rastermap_comp(dff_path,rastermap_save_path)

def visualize_rastermap(filename,segmentation_folder,rastermap_folder):
    f_str=os.path.split(filename)[-1]
    rastermap_f_str=f_str.replace('aligned.h5','rm.npy')
    rois_f_str=f_str.replace('aligned.h5','rois.npy')
    rm_path=os.path.join(os.path.normpath(rastermap_folder),rastermap_f_str)
    roi_path=os.path.join(os.path.normpath(segmentation_folder),rois_f_str)
    run_rm_viz(roi_path,rm_path)

def complete_viz(filename,roi_folder,dff_folder):
    f_str=os.path.split(filename)[-1]
    dff_f_str=f_str.replace('aligned.h5','dff.npy')
    dff_path=os.path.join(os.path.normpath(dff_folder),dff_f_str)
    rois_f_str=f_str.replace('aligned.h5','rois.npy')
    roi_path=os.path.join(os.path.normpath(roi_folder),rois_f_str)
    run_complete_viz(filename,roi_path,dff_path)

def transparent_cells(filename,roi_folder,dff_folder):
    f_str=os.path.split(filename)[-1]
    dff_f_str=f_str.replace('aligned.h5','dff.npy')
    dff_path=os.path.join(os.path.normpath(dff_folder),dff_f_str)
    rois_f_str=f_str.replace('aligned.h5','rois.npy')
    roi_path=os.path.join(os.path.normpath(roi_folder),rois_f_str)
    run_transparent_cells(filename,roi_path,dff_path)

#Detrending
#save_folder_detrending,save_folder_segmentation,save_folder_dff,save_folder_rastermap
filename='//ZMN-HIVE/User-Data/Maria/check_registration/control/fish17_6dpf_medium_aligned.h5'
filename='//ZMN-HIVE/User-Data/Maria/check_registration/amphetamine/fish44_6dpf_amph_aligned.h5'
reg_t_ind=0
save_folder_mask='//ZMN-HIVE/User-Data/Maria/masked'
save_folder_masked='//ZMN-HIVE/User-Data/Maria/masked'
save_folder_detrending='//ZMN-HIVE/User-Data/Maria/detrended'
save_folder_segmentation='//ZMN-HIVE/User-Data/Maria/segmented'
save_folder_dff='//ZMN-HIVE/User-Data/Maria/dff'
save_folder_rastermap='//ZMN-HIVE/User-Data/Maria/rastermap'
if __name__=='__main__':
    #pipeline(filename,0,save_folder_mask,save_folder_masked,0,0,0)
    #masking(filename,reg_t_ind,save_folder_mask)
    #detrending(filename,save_folder_mask,save_folder_detrending)
    #segment(filename,save_folder_detrending,save_folder_segmentation)
    #dff(filename,save_folder_segmentation,save_folder_dff)
    #rastermap_save(filename, save_folder_dff, save_folder_rastermap)
    #visualize_rastermap(filename,save_folder_segmentation,save_folder_rastermap)
    #complete_viz(filename,save_folder_segmentation, save_folder_dff)
    transparent_cells(filename,save_folder_segmentation, save_folder_dff)
