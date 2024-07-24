#import os
#os.environ["CUDA_VISIBLE_DEVICES"]='0'
import numpy as np
import scipy.io
from scipy.io import savemat
import sys

def read_channel_mesh_bin(path, NX, NY, NZ, LX, LZ):

    X = np.arange(NX) * LX / NX
    Y = np.fromfile(f'{path}mesh.bin', dtype='double') + 1
    Z = np.arange(NZ) * LZ / NZ
    
    return X, Y, Z

case = 'A03'
root = '/   /'
filename = root + 'y_predic' + case + '.npy', root + 'y_target' + case + '.npy'

NX = 64
NY = 128
NZ = 64
LX = np.pi
LZ = np.pi / 2
path = '/.../'

X, Y, Z = read_channel_mesh_bin(path, NX, NY, NZ, LX, LZ)

for idx in range(1,int(4000/500)+1):
    
    y_predic = np.load(filename[0])[500*(idx-1):500*idx,:,:,:,:]
    y_target = np.load(filename[1])[500*(idx-1):500*idx,:,:,:,:]

    data = {'x': X, 'y': Y, 'z': Z, 'target': y_target, 'predic': y_predic}
    subcase = case + '_' + str(idx)
    savemat('/ ... /matlabmatrix' + subcase + '.mat', data)
    print(subcase + '.mat has been saved')
