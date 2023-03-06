import numpy as np
import pandas as pd
import h5py
#import openpmd_api

hdf = h5py.File('inception_v3_weights_tf_dim_ordering_tf_kernels.h5','r')

m = open('mean.txt','w')
v = open('variance.txt','w')
b = open('beta.txt','w')
pb = open('Prediction_bias.txt','w') 

pbias = hdf.get('/predictions/predictions/bias:0')

if pbias is not None:
	nppbias = np.array(pbias)
	pbiasList = nppbias.tolist()
	#print(Kernel)
	pbiasString = ",".join(map(str,pbiasList))
	pb.write(pbiasString)


for k in range(94):
	K = str(k+1)
	mean = hdf.get('/batch_normalization_'+K+'/batch_normalization_'+K+'/moving_mean:0')
	if mean is not None:
		npmean = np.array(mean)
		listmean = npmean.tolist()
		#print(Kernel)
		stringmean = ",".join(map(str,listmean))
		m.write(stringmean)
		m.write("\n")

	var = hdf.get('/batch_normalization_'+K+'/batch_normalization_'+K+'/moving_variance:0')
	if var is not None:
		npvar = np.array(var)
		listvar = npvar.tolist()
		#print(Kernel)
		stringvar = ",".join(map(str,listvar))
		v.write(stringvar)
		v.write("\n")

	beta = hdf.get('/batch_normalization_'+K+'/batch_normalization_'+K+'/beta:0')
	if beta is not None:
		npbeta = np.array(beta)
		listbeta = npbeta.tolist()
		#print(Kernel)
		stringbeta = ",".join(map(str,listbeta))
		b.write(stringbeta)
		b.write("\n")
		
b.close()
m.close()
v.close()
pb.close()

hdfKer = h5py.File('inception_v3_weights_tf_dim_ordering_tf_kernels.h5','r')

ker = hdfKer.get('/predictions/predictions/kernel:0')

if ker is not None:
	view = np.array(ker)
	np.savetxt("predictionKernel.txt", view, delimiter = ',')

for l in range(94):
	L = str(l+1)
	f = open('conv2d'+L+'.txt','w')	
	ck = hdf.get('/conv2d_'+L+'/conv2d_'+L+'/kernel:0')
	if ck is not None:
		npck = np.array(ck)
		flatck = npck.flatten()
		listck = flatck.tolist()
		stringck = ",".join(map(str,listck))
		f.write(stringck)
	f.close()
		
