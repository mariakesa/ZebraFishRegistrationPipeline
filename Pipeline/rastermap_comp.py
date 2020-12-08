from rastermap import Rastermap
import numpy as np

def rastermap_comp(dff_path, rastermap_save_path):
    dff=np.load(dff_path)
    model = Rastermap(n_components=1, n_X=100).fit(dff)
    np.save(rastermap_save_path,model.embedding[:,0])
