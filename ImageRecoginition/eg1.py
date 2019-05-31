import numpy as np
import matplotlib.pyplot as plt
import scipy
import lr_utils as util
from PIL import Image
from scipy import ndimage

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = util.load_dataset()

print(train_set_y)

index = 24
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
plt.imshow(train_set_x_orig[index])
plt.show()

def initialize_parameters_deep(layer_dims):
  np.random.seed(3)
  parameters = {}
  L = len(layer_dims)            # number of layers in the network
  for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dim[l],layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        #assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        #assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
  return parameters