import numpy as np
import matplotlib as mpl
import matplotlib.path as mpath
from matplotlib import rc
import matplotlib.pyplot as plt
import netCDF4 as nc 
import pandas as pd 
import xarray as xr
from xgcm import Grid
from os.path import join,expanduser
import ecco_v4_py as ecco
import seaborn as sns
import tensorflow as tf
import scipy.io
import time
import warnings
import matplotlib.colors as colors
import dask
import statsmodels.api as sm
from pyDOE import *

np.random.seed(4321)
tf.set_random_seed(4321)

# LOADING IN ECCO GRID + DATA

grid_path = '/projects/SOCCOM/datasets/ecco/Version4/Release4/nctiles_grid/'
ecco_grid = xr.open_dataset(grid_path+'ECCO-GRID.nc')

ds = xr.Dataset()
tags = ['THETA','SALT','EVEL', 'NVEL', 'WVELMASS']

for i in range(1992,1993):
    for tag in tags:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            data_path = '/projects/SOCCOM/datasets/ecco/Version4/Release4/interp_monthly/'+tag+'/' + str(i) +'/'+tag+'_' + str(i) + '_??.nc'
            dsnow = xr.open_mfdataset(data_path,chunks="auto",data_vars='minimal',coords='minimal', compat='override') 
            ds = xr.merge([ds, dsnow])
    print('the year ' + str(i) + ' is loaded')
    
# download the bolus streamfunction
with dask.config.set(**{'array.slicing.split_large_chunks': True}):    
    data_path = '/projects/CDEUTSCH/DATA/OCEAN_BOLUS_STREAMFUNCTION_ECCO_V4r4_latlon_0p50deg.nc'
    dsnow = xr.open_mfdataset(data_path,chunks="auto",data_vars='minimal',coords='minimal', compat='override') 

# download the diffusivity coefficients
with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    data_path = '/projects/CDEUTSCH/DATA/OCEAN_3D_MIXING_COEFFS_ECCO_V4r4_latlon_0p50deg.nc'
    dsnow2 = xr.open_mfdataset(data_path,chunks="auto",data_vars='minimal',coords='minimal', compat='override') 
ds2 = xr.merge([dsnow, dsnow2])

# reindex using i,j,k instead of lat lon depth
ds2 = ds2.assign_coords(i=('longitude', np.arange(0,720)))
ds2 = ds2.assign_coords(j=('latitude', np.arange(0,360)))
ds2 = ds2.assign_coords(k=('Z', np.arange(0,50)))
ds2 = ds2.swap_dims({"longitude":"i","latitude":"j","Z":"k"})

ds = xr.merge([ds, ds2])
del ds2, dsnow, dsnow2

Sx = ds.PsiX_interp/ds.KAPGM
Sy = ds.PsiY_interp/ds.KAPGM
Smag = np.sqrt(Sx**2 + Sy**2)
Kx = ds.KAPREDI*(1+2*Sx)
Ky = ds.KAPREDI*(1+2*Sy)
Kz = ds.KAPREDI*(Smag**2)
ds = xr.merge([ds, Kx.rename('Kx'), Ky.rename('Ky'), Kz.rename('Kz')])
del Sx, Sy, Smag, Kx, Ky, Kz

# make the XGCM object
xgcm_grid = ecco.get_llc_grid(ecco_grid)

# interpolate w velocity onto regular grid
WVELMASS_interp = xgcm_grid.interp(ds.WVELMASS, "Z", boundary='fill')
# adjusting the pressure anomaly to not be 1/rho
# PHI = ds.PHIHYD*(1029) # rho_const = 1029
# creating land mask
maskC = ds.SALT.where(np.logical_or(ds.SALT.isnull(), ds.SALT==0),1).isel(time=0)

# get the grid
ds = xr.merge([ds, WVELMASS_interp.rename('WVELMASS_interp'), maskC.rename('maskC')],compat='override')

# DATA PREPROCESSING (with Navier Stokes normalization)

# gets the index of the array for each dimension before flattening
def using_multiindex(A, columns):
    shape = A.shape
    index = pd.MultiIndex.from_product([range(s) for s in shape], names=columns)
    df = pd.DataFrame({'A': A.flatten()}, index=index).reset_index()
    return df

def make_nninputs(ds, tags, time_ind, depth_ind, j_ind, i_ind, N_train, noise):
    # ds is the main ecco data array
    # tags is a list of strings corresponding to the input variables i.e. ['THETA','SALT', ...]
        # first tagged variable must be in 4d, others can be in 3d
    # tile_ind, depth_ind, j_ind, and i_ind are tuples for subsetted data
    # N_train is a number between 0 and 100 describing percentage of data sampled
    # noise is the percent Gaussian noise you would like to incorporate, must be integer, ex: 1 for 1% Gaussian noise
    # returns inputs_nn which is an array of all the training data of size N points x [time, depth, lat, lon, tags...]
    # returns coords_nn which is an array of all the training data coordinates of size N points x [time, k, j, i]
    
    subset_template = ds[tags[0]].where(ds.maskC>0).isel(time=slice(*time_ind),k=slice(*depth_ind),j=slice(*j_ind),i=slice(*i_ind))
    # assuming that only one tile is chosen (OTHERWISE WILL HAVE TO CHANGE CODE)
    inputs_nn = using_multiindex(subset_template.squeeze().values, ['time', 'k', 'j', 'i']) # makes array N x 5 [time, k, j, i, *first tag*]
    inputs_nn = inputs_nn.to_numpy()
    if N_train < 100: # for validation data no need to scramble coordinates
        N_train = int(N_train/100*len(inputs_nn[:,0]))
        train_idx = np.random.choice(len(inputs_nn[:,0]), N_train, replace=False) # Generate a random sample from np.arange(N*T) of size N_train
    else: 
        train_idx = range(len(inputs_nn[:,0]))
    inputs_nn = inputs_nn[train_idx,:] # subsample the training data
    coords_nn = inputs_nn[:,0:4].astype(int) # makes the time,kji indices integers for future use
    inputs_nn = inputs_nn[:,4:] # trim array by cutting out coordinates
    
    for tag in tags:
        throwaway = ds[tag].where(ds.maskC).isel(time=slice(*time_ind),k=slice(*depth_ind),j=slice(*j_ind),i=slice(*i_ind)).squeeze().values.flatten()[train_idx,np.newaxis]
        if noise>0:
            throwaway = throwaway + noise/100*np.std(throwaway)*np.random.randn(throwaway.shape[0], throwaway.shape[1])
        inputs_nn = np.concatenate((inputs_nn, throwaway),axis=1)

    inputs_nn = inputs_nn[:,1:] # need to remove the first dummy variable
    
    # assuming that only one tile is chosen (OTHERWISE WILL HAVE TO CHANGE CODE)

    a=6371*1000 # radius of the earth in meters
    throwaway = np.radians(ds.longitude.isel(i=slice(*i_ind)))*a*np.cos(np.radians(ds.latitude.isel(j=slice(*j_ind))))
    throwaway = throwaway.values
    throwaway2 = throwaway[coords_nn[:,3],coords_nn[:,2]][:,np.newaxis]  
    inputs_nn = np.concatenate((throwaway2,inputs_nn),axis=1)
    throwaway = np.radians(ds.latitude.isel(j=slice(*j_ind)))*a*np.cos(0*np.radians(ds.longitude.isel(i=slice(*i_ind)))) # these are the Y positions lat*r
    throwaway = throwaway.values
    throwaway2 = throwaway[coords_nn[:,2],coords_nn[:,3]][:,np.newaxis]
    inputs_nn = np.concatenate((throwaway2,inputs_nn),axis=1)
    throwaway = ds.Z.isel(k=slice(*depth_ind)).values[coords_nn[:,1],np.newaxis] # get depth coordinates
    inputs_nn = np.concatenate((throwaway,inputs_nn),axis=1)
    throwaway = ds.timestep.isel(time=slice(*time_ind)).values[coords_nn[:,0],np.newaxis]*3600
    inputs_nn = np.concatenate((throwaway,inputs_nn),axis=1)

    lb_nn = np.nanmin(inputs_nn,axis=0)
    char_nn = (np.nanmax(inputs_nn,axis=0)-lb_nn)/2
    inputs_nn = (inputs_nn-lb_nn)/char_nn - 1
    
    # get rid of the nans after calculating mean + std
    inputs_nn[np.isnan(inputs_nn)] = 0
                                   
    return subset_template, inputs_nn, coords_nn, char_nn, lb_nn

def split_nninputs(inputs_nn, coords_nn):
    
    # splits a full training dataset into a 20% validation and 80% training portion
    
    N_train = int(0.2*len(inputs_nn[:,0])) # 20% data used for validation
    val_idx = np.random.choice(len(inputs_nn[:,0]), N_train, replace=False)
    # select out 20% of training data for validation
    inputs_val = inputs_nn[val_idx,:]
    coords_val = coords_nn[val_idx,:]
    # leave remaining 80% of training data for training
    inputs_train = np.delete(inputs_nn, val_idx, axis=0)
    coords_train = np.delete(coords_nn, val_idx, axis=0)
    
    return inputs_val, coords_val, inputs_train, coords_train

# # MAKE TESTING, VALIDATION, AND TRAINING DATA

# tags = ['THETA','SALT','EVEL', 'NVEL','WVELMASS_interp','PHI','RHOAnoma']

time_ind = (0,12)
k_ind = (0,20)
j_ind = (40,90)
i_ind = (300,350)

# testing dataset
with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    subset_template, inputs_test, coords_test, char_test, lb_test = make_nninputs(ds,tags, time_ind, k_ind, j_ind, i_ind,100,0) 
print('testing dataset is processed!')

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    _, inputs_train, coords_train, char_train, lb_train = make_nninputs(ds,tags, time_ind, k_ind, j_ind, i_ind, 5, 0)
print('training dataset is processed!')

inputs_val, coords_val, inputs_train, coords_train = split_nninputs(inputs_train, coords_train)
print('validation dataset is processed!')

# calculate the coriolis constant for the testing points
a=6371*1000 # radius of the earth
throwaway = (inputs_test[:,2]+1)*char_test[2] + lb_test[2]
f_test = 2*(2*np.pi)/86164*np.sin(np.radians(throwaway*180/np.pi/a))
f_test = f_test.reshape((f_test.shape[0],1)).astype('float32')

# NEW NEW VERSION WITH KX, KY, AND KZ AS OUTPUTS
class PhysicsInformedNN:
    # Initialize the class
    
    def __init__(self, inputs_nn, char_nn, lb_nn, col_nn, val_nn, layers, f_nn, gamma):
        
        # inputs_nn are the normalized training data
        # char_nn are the characteristic scales of the unnormalized training data
        # lb_nn are the minima of the unnormalized training data
        # col_nn are the normalized collocation points
        # layers are the structure of the PINN
        # f_nn is the coriolis constant for the collocation points
        # gamma is a hyperparameter whose value ranges from 0 to 1 to weigh the overall loss: (1-gamma)*data_loss + gamma*eq_loss, must be float (0.0 instead of 0)

        # Coriolis constants
        self.f_nn = f_nn
        
        # equation weighting
        self.gamma = tf.constant(gamma)
        
        X = inputs_nn[:,0:4]
                
        self.X = X
        
        self.char_nn = char_nn
        self.lb_nn = lb_nn
        
        # Training points
        
        self.t = inputs_nn[:,0:1]
        self.k = inputs_nn[:,1:2]
        self.j = inputs_nn[:,2:3]
        self.i = inputs_nn[:,3:4]
    
        self.T = inputs_nn[:,4:5]
        self.S = inputs_nn[:,5:6]
        # self.u = inputs_nn[:,6:7]
        # self.v = inputs_nn[:,7:8]
        # self.w = inputs_nn[:,8:9]
        # self.p = inputs_nn[:,9:10]
        # self.rho = inputs_nn[:,10:11]
        
        # Collocation points
        
        self.t_c = col_nn[:,0:1]
        self.k_c = col_nn[:,1:2]
        self.j_c = col_nn[:,2:3]
        self.i_c = col_nn[:,3:4]
        
        self.u_c = col_nn[:,6:7]
        self.v_c = col_nn[:,7:8]
        self.w_c = col_nn[:,8:9]
        
        # Validation points
        
        self.t_v = val_nn[:,0:1]
        self.k_v = val_nn[:,1:2]
        self.j_v = val_nn[:,2:3]
        self.i_v = val_nn[:,3:4]
        
        self.T_v = val_nn[:,4:5]
        self.S_v = val_nn[:,5:6]
#         self.u_v = val_nn[:,6:7]
#         self.v_v = val_nn[:,7:8]
#         self.w_v = val_nn[:,8:9]
#         self.p_v = val_nn[:,9:10]
#         self.rho_v = val_nn[:,10:11]
        
        self.layers = layers
        
        # Counter for the reported NS terms
        self.counter = 0
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # training point tf placeholders
        self.i_tf = tf.placeholder(tf.float32, shape=[None, self.i.shape[1]])
        self.j_tf = tf.placeholder(tf.float32, shape=[None, self.j.shape[1]])
        self.k_tf = tf.placeholder(tf.float32, shape=[None, self.k.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.S_tf = tf.placeholder(tf.float32, shape=[None, self.S.shape[1]])
        self.T_tf = tf.placeholder(tf.float32, shape=[None, self.T.shape[1]])

        # self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        # self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]]) 
        # self.w_tf = tf.placeholder(tf.float32, shape=[None, self.w.shape[1]])
        # self.p_tf = tf.placeholder(tf.float32, shape=[None, self.p.shape[1]])
        # self.rho_tf = tf.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
        
        # collocation point tf placeholders
        self.i_ctf = tf.placeholder(tf.float32, shape=[None, self.i_c.shape[1]])
        self.j_ctf = tf.placeholder(tf.float32, shape=[None, self.j_c.shape[1]])
        self.k_ctf = tf.placeholder(tf.float32, shape=[None, self.k_c.shape[1]])
        self.t_ctf = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])
        
        self.u_ctf = tf.placeholder(tf.float32, shape=[None, self.u_c.shape[1]])
        self.v_ctf = tf.placeholder(tf.float32, shape=[None, self.v_c.shape[1]])
        self.w_ctf = tf.placeholder(tf.float32, shape=[None, self.w_c.shape[1]])
        
        # validation tf placeholders
        self.i_vtf = tf.placeholder(tf.float32, shape=[None, self.i_v.shape[1]])
        self.j_vtf = tf.placeholder(tf.float32, shape=[None, self.j_v.shape[1]])
        self.k_vtf = tf.placeholder(tf.float32, shape=[None, self.k_v.shape[1]])
        self.t_vtf = tf.placeholder(tf.float32, shape=[None, self.t_v.shape[1]])
        
        self.S_vtf = tf.placeholder(tf.float32, shape=[None, self.S_v.shape[1]])
        self.T_vtf = tf.placeholder(tf.float32, shape=[None, self.T_v.shape[1]])

        # training points
        self.T_pred, self.S_pred, self.Kx, self.Ky, self.Kz = self.net_NS(self.t_tf, self.k_tf, self.j_tf, self.i_tf)

        # validation points
        self.T_val_pred, self.S_val_pred, _, _, _ = self.net_NS(self.t_vtf, self.k_vtf, self.j_vtf, self.i_vtf)

        # collocation points
        self.fT_pred, self.fS_pred = self.net_f_NS(self.t_ctf, self.k_ctf, self.j_ctf, self.i_ctf)
        
        # losses
        
        self.fT_pred = tf.reduce_mean(tf.square(self.fT_pred))
        self.fS_pred = tf.reduce_mean(tf.square(self.fS_pred)) 
        
        # self.diffloss = tf.math.maximum(-1*self.Kx,0) + tf.math.maximum(-1*self.Ky,0) + tf.math.maximum(-1*self.Kz,0)
        
        self.eqloss = self.fT_pred + self.fS_pred
                    # tf.reduce_mean(tf.square(self.fT_pred)) + \
                    # tf.reduce_mean(tf.square(self.fS_pred)) 

        self.dataloss = tf.reduce_mean(tf.square(self.T_tf - self.T_pred)) + \
                    tf.reduce_mean(tf.square(self.S_tf - self.S_pred))
        
        self.valloss = tf.reduce_mean(tf.square(self.T_vtf - self.T_val_pred)) + \
                    tf.reduce_mean(tf.square(self.S_vtf - self.S_val_pred)) 
        
        # self.wloss = tf.reduce_mean(1.0 - tf.math.exp(-tf.square(self.w_pred)/0.001))

        self.loss = (1-self.gamma)*self.dataloss + self.gamma*(self.eqloss) #+ self.diffloss
    
        # saved losses
        
        self.losses = []
        self.eq_losses = []
        self.data_losses = []
        self.val_losses = []
        self.fT_pred_losses = []
        self.fS_pred_losses = []
        # self.fu_pred_losses = []
        # self.u_t_losses = []
        # self.uu_x_losses = []
        # self.vu_y_losses = []
        # self.wu_z_losses = []
        # self.fv_losses = []
        # self.p_x_losses = []
        # self.u_pred_losses = []
        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 2000,  # original maxiter is 50000
                                                                           'maxfun': 50000,
                                                                           'maxcor': 25,
                                                                           'maxls': 25,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) 
        
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init)

    def initialize_NN(self, layers): 
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        # initializer = tf.variance_scaling_initializer(scale=1.0, mode='fan_in')
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
        # return tf.Variable(initializer(shape=(in_dim, out_dim)), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            # H = tf.keras.activations.selu(tf.add(tf.matmul(H, W), b))
            # H = tf.tanh(tf.add(tf.matmul(H, W), b))
            H = tf.sin(tf.add(tf.matmul(H, W), b)) # prediction ability not as good
            # H = tf.nn.relu(tf.add(tf.matmul(H, W), b)) # prediction ability not as good
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b) # linear activation function for the last layer
        return Y
        
    def net_f_NS(self, t, k, j, i):
        
        u_hat = self.u_c
        v_hat = self.v_c
        w_hat = self.w_c
        # getting equation losses on collocation points
        
        with tf.GradientTape(persistent=True) as tape2:
            # Watch the input variables
            tape2.watch(t)
            tape2.watch(k)
            tape2.watch(j)
            tape2.watch(i)
        
            with tf.GradientTape(persistent=True) as tape:
                # Watch the input variables
                tape.watch(t)
                tape.watch(k)
                tape.watch(j)
                tape.watch(i)

                uvwp = self.neural_net(tf.concat([t, k, j, i],1),self.weights,self.biases)
                T_hat = uvwp[:,0:1]
                S_hat = uvwp[:,1:2]
                Kx = uvwp[:,2:3]
                Ky = uvwp[:,3:4]
                Kz = uvwp[:,4:5]
                
            T_t, T_z, T_y, T_x = tape.gradient(T_hat, [t, k, j, i])
            S_t, S_z, S_y, S_x = tape.gradient(S_hat, [t, k, j, i])
            Kx_x = tape.gradient(Kx, i)
            Ky_y = tape.gradient(Ky, j)
            Kz_z = tape.gradient(Kz, k)
        
        char_nn = self.char_nn
        lb_nn = self.lb_nn

        f = self.f_nn
       
        T_zz = tape2.gradient(T_z, k)
        T_yy = tape2.gradient(T_y, j)
        T_xx = tape2.gradient(T_x, i)
        S_zz = tape2.gradient(S_z, k)
        S_yy = tape2.gradient(S_y, j)
        S_xx = tape2.gradient(S_x, i)
        
        u_full = (u_hat+1)*char_nn[6]+lb_nn[6]
        v_full = (v_hat+1)*char_nn[7]+lb_nn[7]
        w_full = (w_hat+1)*char_nn[8]+lb_nn[8]
        # w_full = w_hat
        # rho_full = (rho_hat+1)*char_nn[10]+lb_nn[10]
    
        # temperature re conservation
        fT = char_nn[3]/char_nn[4]*(char_nn[4]/char_nn[0]*T_t + (char_nn[4]/char_nn[3]*u_full - Kx_x/char_nn[3])*T_x + (char_nn[4]/char_nn[2]*v_full - Ky_y/char_nn[2])*T_y + (char_nn[4]/char_nn[1]*w_full - Kz_z/char_nn[1])*T_z - Kx*char_nn[4]/char_nn[3]**2*T_xx - Kx*char_nn[4]/char_nn[2]**2*T_yy - Kz*char_nn[4]/char_nn[1]**2*T_zz) 
        
        # salinity conservation
        fS = char_nn[3]/char_nn[5]*(char_nn[5]/char_nn[0]*T_t + (char_nn[5]/char_nn[3]*u_full - Kx_x/char_nn[3])*S_x + (char_nn[5]/char_nn[2]*v_full - Ky_y/char_nn[2])*S_y + (char_nn[5]/char_nn[1] - Kz_z/char_nn[1])*w_full*S_z - Kx*char_nn[5]/char_nn[3]**2*S_xx - Kx*char_nn[5]/char_nn[2]**2*S_yy - Kz*char_nn[5]/char_nn[1]**2*S_zz) 
        
        del tape, tape2
        
        return fT, fS
    
    def net_NS(self, t, k, j, i):
        
        # getting data loss on training points
        
        uvwp = self.neural_net(tf.concat([t, k, j, i], 1), self.weights, self.biases)
        T_hat = uvwp[:,0:1]
        S_hat = uvwp[:,1:2]
        Kx_hat = uvwp[:,2:3]
        Ky_hat = uvwp[:,3:4]
        Kz_hat = uvwp[:,4:5]
        
        return T_hat, S_hat, Kx_hat, Ky_hat, Kz_hat

    def callback(self, loss, dataloss, eqloss, valloss, fT_pred, fS_pred, Kz, Ky, Kx):
    # def callback(self, loss, dataloss, eqloss, fu_pred):
        
        callback_val = tf.keras.callbacks.EarlyStopping(monitor='valloss', patience=10, mode='auto',baseline=None)
        
        if self.counter % 10 == 0:
            print('L-BFGS-B It: %d, Loss: %.3e, Data loss: %.3e, Eq loss: %.3e, Val loss: %.3e, Kx: %.3e, Ky: %.3e, Kz: %.3e' % (self.counter, loss, dataloss, eqloss, valloss, Kx.mean(), Ky.mean(), Kz.mean()))
            self.losses.append(loss)
            self.eq_losses.append(eqloss)
            self.data_losses.append(dataloss)
            self.val_losses.append(valloss)
            self.fT_pred_losses.append(fT_pred)
            self.fS_pred_losses.append(fS_pred)
        
        self.counter = self.counter + 1
        return callback_val
 
    def train(self, nIter): 

        tf_dict = {self.i_tf: self.i, self.j_tf: self.j, self.k_tf: self.k, self.t_tf: self.t, self.S_tf: self.S, self.T_tf: self.T,\
                   self.i_ctf: self.i_c, self.j_ctf: self.j_c, self.k_ctf: self.k_c, self.t_ctf: self.t_c, self.u_ctf: self.u_c, self.v_ctf: self.v_c, self.w_ctf: self.w_c,\
                  self.i_vtf: self.i_v, self.j_vtf: self.j_v, self.k_vtf: self.k_v, self.t_vtf: self.t_v, self.S_vtf: self.S_v, self.T_vtf: self.T_v}#, self.u_vtf: self.u_v, self.v_vtf: self.v_v, self.w_vtf: self.w_v}
        
        start_time = time.time()
        
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                
                loss_value = self.sess.run(self.loss, tf_dict)
                data_value = self.sess.run(self.dataloss, tf_dict)
                eq_value = self.sess.run(self.eqloss, tf_dict)
                val_value = self.sess.run(self.valloss, tf_dict)
                # tracer concentration values
                fT_value = self.sess.run(self.fT_pred, tf_dict)
                fS_value = self.sess.run(self.fS_pred, tf_dict)
                # fT_value = np.mean(np.square(self.sess.run(self.fT_pred,tf_dict)))
                # fS_value = np.mean(np.square(self.sess.run(self.fS_pred,tf_dict)))
                Kz_value = np.mean(self.sess.run(self.Kz, tf_dict))
                Ky_value = np.mean(self.sess.run(self.Ky, tf_dict))
                Kx_value = np.mean(self.sess.run(self.Kx, tf_dict))
                # w_value = self.sess.run(self.w_diff, tf_dict)
                # fS_value = self.sess.run(self.fS_pred, tf_dict)
                elapsed = time.time() - start_time
                
                print('Adam It: %d, Total loss: %.3e, Data loss: %.3e, Eq loss: %.3e, Val loss: %.3e, Kx: %.3e, Ky: %.3e, Kz: %.2f' % 
                      (it, loss_value, data_value, eq_value, val_value, fT_value, fS_value, elapsed))

                self.losses.append(loss_value)
                self.eq_losses.append(eq_value)
                self.data_losses.append(data_value)
                self.val_losses.append(val_value)
                self.fT_pred_losses.append(fT_value)
                self.fS_pred_losses.append(fS_value)

        self.counter = nIter
                
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                # fetches = [self.loss, self.dataloss, self.eqloss, self.fu_pred],
                                fetches = [self.loss, self.dataloss, self.eqloss, self.valloss, self.fT_pred, self.fS_pred, self.Kz, self.Ky, self.Kx],
#                                 loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='auto',verbose=2,baseline=None))
                                loss_callback = self.callback)
        
        # return self.losses, self.eq_losses, self.data_losses, self.fu_pred_losses
        return self.losses, self.eq_losses, self.data_losses, self.val_losses, self.fT_pred_losses, self.fS_pred_losses
    
    def predict(self, t_star, k_star, j_star, i_star):
        
        tf_dict = {self.i_tf: i_star, self.j_tf: j_star, self.k_tf: k_star, self.t_tf: t_star}

        T_star = self.sess.run(self.T_pred, tf_dict)
        S_star = self.sess.run(self.S_pred, tf_dict)
        Kx_star = self.sess.run(self.Kx, tf_dict)
        Ky_star = self.sess.run(self.Ky, tf_dict)
        Kz_star = self.sess.run(self.Kz, tf_dict)

        return T_star, S_star, Kx_star, Ky_star, Kz_star

    def compute_gradients(self, t, k, j, i):
        
        # check that the automatic differentiation matches the numerical differentiation outside of the PINN
        
        tf_dict = {self.i_tf: i, self.j_tf: j, self.k_tf: k, self.t_tf: t}

        with tf.GradientTape(persistent=True) as tape:
            # Watch the input variables
            tape.watch(self.t_tf)
            tape.watch(self.k_tf)
            tape.watch(self.j_tf)
            tape.watch(self.i_tf)

            uvwp = self.neural_net(tf.concat([self.t_tf, self.k_tf, self.j_tf, self.i_tf], 1), self.weights, self.biases)
            T_hat = uvwp[:,0:1]
            S_hat = uvwp[:,1:2]
            Kx = uvwp[:,2:3]
            Ky = uvwp[:,3:4]
            Kz = uvwp[:,4:5]
            
        inputs = [self.t_tf, self.k_tf, self.j_tf, self.i_tf]

        # Compute the gradients 
        gradients = [self.sess.run(tape.gradient(u_hat, input_var),feed_dict=tf_dict) for input_var in inputs]

        # Clean up the tape
        del tape

        return gradients, self.sess.run(u_hat,feed_dict=tf_dict)
    
    def save_model(self, save_path):
        self.saver.save(self.sess, save_path)
        print('model saved!')
        
if __name__ == "__main__": 
    
    layers = [4, 20, 20, 20, 20, 5] # first layer should be size of X in the PINN

    model = PhysicsInformedNN(inputs_train, char_train, lb_train, inputs_test, inputs_val, layers, f_test, 0.1)
    nIter = 500 # number of Adams optimizer training iterations
    losses, eq_losses, data_losses, val_losses, temp_losses, salt_losses = model.train(nIter)

# SAVING THE MODEL
ckpt_file = "/scratch/gpfs/wc4720/SOCCOM/saved-models/diffusivity-noprho/latlon_k020_j4090_i300350_iter2500_layer20_gamma10_omitvel2.ckpt"
array_file = "/scratch/gpfs/wc4720/SOCCOM/saved-models/diffusivity-noprho/latlon_k020_j4090_i300350_iter2500_layer20_gamma10_omitvel2.npz"

model.save_model(ckpt_file)
np.savez(array_file, inputs_train=inputs_train, char_train=char_train, lb_train=lb_train, f_test = f_test, inputs_val = inputs_val, coords_val = coords_val, inputs_test = inputs_test, char_test = char_test, lb_test = lb_test, subset_template = subset_template, layers=layers, coords_test=coords_test, data_losses=data_losses, eq_losses=eq_losses, val_losses = val_losses, temp_losses = temp_losses, salt_losses = salt_losses)
