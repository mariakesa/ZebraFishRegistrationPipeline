from mask_gui import run_mask_gui
import os

def pipeline(filename,reg_t_ind,save_folder_mask,save_folder_detrending,save_folder_segmentation,save_folder_dff,save_folder_rastermap):
    f_str=os.path.split(filename)[-1]
    print(filename)
    mask_f_str=f_str.replace('aligned.h5','mask.npy')
    masked_f_str=f_str.replace('aligned.h5','masked.h5')
    save_mask_filename=os.path.join(os.path.normpath(save_folder_mask),mask_f_str)
    save_masked_filename=os.path.join(os.path.normpath(save_folder_masked),masked_f_str)
    run_mask_gui(filename,save_masked_filename,save_mask_filename,reg_t_ind)

filename='//ZMN-HIVE/User-Data/Maria/check_registration/control/fish17_6dpf_medium_aligned.h5'
save_folder_mask='//ZMN-HIVE/User-Data/Maria/masked'
save_folder_masked='//ZMN-HIVE/User-Data/Maria/masked'
pipeline(filename,0,save_folder_mask,save_folder_masked,0,0,0)
