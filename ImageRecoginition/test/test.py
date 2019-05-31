import numpy as np
import h5py
import matplotlib.image as img
m=img.imread('1.png')


with h5py.File('new.h5',"w") as hdf:

  hdf.create_dataset('dataset1',data=m )
  
data=h5py.File("new.h5","r")
test_set = np.array(data) # your test set features

