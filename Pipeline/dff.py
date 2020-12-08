import numpy as np

def compute_dff(traces_path,save_dff_path):
  traces=np.load(traces_path)
  F0=np.mean(traces[:,250:350],axis=1)
  dff=(traces-F0[:,None])/F0[:,None]
  np.save(save_dff_path,dff)
