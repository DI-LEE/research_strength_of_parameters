import numpy as np
import pandas as pd
import h5py
#import openpmd_api

hdf = h5py.File('copy.h5','a')
f = open('shiftbeta.txt','r')
f2 = open('shiftmean.txt','r')
f3 = open('shiftvariance.txt','r')
f4 = open('shiftPrediction_bias.txt','r')
for k in range(94):
	K = str(k+1)
	kernel3 = hdf.get('/batch_normalization_'+K+'/batch_normalization_'+K+'/beta:0')
	kernel4 = hdf.get('/batch_normalization_'+K+'/batch_normalization_'+K+'/moving_mean:0')
	kernel5 = hdf.get('/batch_normalization_'+K+'/batch_normalization_'+K+'/moving_variance:0')
	if kernel3 is not None:
		del hdf['/batch_normalization_'+K+'/batch_normalization_'+K+'/beta:0']
		string = f.readline()
		new_str = string.replace("\n", "")
		list_a = new_str.split(',')
		npy = np.array(list_a)
		floatnpy = np.asarray(npy, dtype = float)
		print(floatnpy)
		hdf['/batch_normalization_'+K+'/batch_normalization_'+K+'/beta:0'] = floatnpy
	if kernel4 is not None:
		del hdf['/batch_normalization_'+K+'/batch_normalization_'+K+'/moving_mean:0']
		string = f2.readline()
		new_str = string.replace("\n", "")
		list_a = new_str.split(',')
		npy = np.array(list_a)
		floatnpy = np.asarray(npy, dtype = float)
		print(floatnpy)
		hdf['/batch_normalization_'+K+'/batch_normalization_'+K+'/moving_mean:0'] = floatnpy
	if kernel5 is not None:
		del hdf['/batch_normalization_'+K+'/batch_normalization_'+K+'/moving_variance:0']
		string = f3.readline()
		new_str = string.replace("\n", "")
		list_a = new_str.split(',')
		npy = np.array(list_a)
		floatnpy = np.asarray(npy, dtype = float)
		print(floatnpy)
		hdf['/batch_normalization_'+K+'/batch_normalization_'+K+'/moving_variance:0'] = floatnpy

kernel6 = hdf.get('/predictions/predictions/bias:0')
if kernel6 is not None:
	del hdf['/predictions/predictions/bias:0']
	string = f4.readline()
	new_str = string.replace("\n", "")
	list_a = new_str.split(',')
	npy = np.array(list_a)
	floatnpy = np.asarray(npy, dtype = float)
	print(floatnpy)
	hdf['/predictions/predictions/bias:0'] = floatnpy

arr = np.array([[]])
list_a = []
f5 = open('shiftPrediction_kernel.txt','r')
kernel6 = hdf.get('/predictions/predictions/kernel:0')

if kernel6 is not None:
	del hdf['/predictions/predictions/kernel:0']
	for j in range(2048):
		string = f5.readline()
		new_str = string.replace("\n","")
		temp = new_str.split(',')
		list_a.append(temp)
npy = np.array(list_a)
floatnpy = np.asarray(npy,dtype = float)
hdf['/predictions/predictions/kernel:0'] = floatnpy


for l in range(94):
	L = str(l+1)
	og = h5py.File('inception_v3_weights_tf_dim_ordering_tf_kernels.h5','r')
	ck = og.get('/conv2d_'+L+'/conv2d_'+L+'/kernel:0')
	cp = h5py.File('copy.h5','a')
	f6 = open('shift'+L+'.txt','r')
	del cp['/conv2d_'+L+'/conv2d_'+L+'/kernel:0']
	npck = np.array(ck)
	string = f6.readline()
	new_str = string.replace("\n","")
	list_a = new_str.split(',')
	npy = np.array(list_a)
	floatnpy = np.asarray(npy, dtype = float)
	npyfloat = floatnpy.reshape(npck.shape)
	cp['/conv2d_'+L+'/conv2d_'+L+'/kernel:0'] = npyfloat









