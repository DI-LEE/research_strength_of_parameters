import numpy as np
import pandas as pd
import h5py
#import openpmd_api

shift = h5py.File('20.h5','a')
hdf = h5py.File('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5','r')

npShift = np.loadtxt('other/shiftConv1.txt')
OGshape = hdf.get('/Conv1/Conv1/kernel:0')
newNpy = npShift.reshape(OGshape.shape)

del shift['/Conv1/Conv1/kernel:0']
shift['/Conv1/Conv1/kernel:0'] = newNpy

npShift = np.loadtxt('other/shiftConv_1.txt')
OGshape = hdf.get('/Conv_1/Conv_1/kernel:0')
newNpy = npShift.reshape(OGshape.shape)

del shift['/Conv_1/Conv_1/kernel:0']
shift['/Conv_1/Conv_1/kernel:0'] = newNpy

#######################################################

npShift = np.loadtxt('other/shiftConv_1_bn_beta.txt')
del shift['/Conv_1_bn/Conv_1_bn/beta:0']
shift['/Conv_1_bn/Conv_1_bn/beta:0'] = npShift

npShift = np.loadtxt('other/shiftConv_1_bn_mean.txt')
del shift['/Conv_1_bn/Conv_1_bn/moving_mean:0']
shift['/Conv_1_bn/Conv_1_bn/moving_mean:0'] = npShift

npShift = np.loadtxt('other/shiftConv_1_bn_var.txt')
del shift['/Conv_1_bn/Conv_1_bn/moving_variance:0']
shift['/Conv_1_bn/Conv_1_bn/moving_variance:0'] = npShift

npShift = np.loadtxt('other/shiftConv_1_bn_gam.txt')
del shift['/Conv_1_bn/Conv_1_bn/gamma:0']
shift['/Conv_1_bn/Conv_1_bn/gamma:0'] = npShift

#######################################################

npShift = np.loadtxt('other/shiftLogits_bias.txt')
del shift['/Logits/Logits/bias:0']
shift['/Logits/Logits/bias:0'] = npShift

npShift = np.loadtxt('other/shiftLogits_kernel.txt')
OGshape = hdf.get('/Logits/Logits/kernel:0')
newNpy = npShift.reshape(OGshape.shape)

del shift['/Logits/Logits/kernel:0']
shift['/Logits/Logits/kernel:0'] = newNpy

####################################################### 

npShift = np.loadtxt('other/shiftbn0_conv_0_bn_depthwise_beta.txt')
del shift['/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/beta:0']
shift['/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/beta:0'] = npShift

npShift = np.loadtxt('other/shiftbn0_conv_0_bn_depthwise_gamma.txt')
del shift['/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/gamma:0']
shift['/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/gamma:0'] = npShift

npShift = np.loadtxt('other/shiftbn0_conv_0_bn_depthwise_mean.txt')
del shift['/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/moving_mean:0']
shift['/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/moving_mean:0'] = npShift

npShift = np.loadtxt('other/shiftbn0_conv_0_bn_depthwise_variance.txt')
del shift['/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/moving_variance:0']
shift['/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/moving_variance:0'] = npShift

#######################################################   !!!!!

npShift = np.loadtxt('other/shiftbn0_conv_0_bn_project_beta.txt')
del shift['/bn0_conv_0_bn_project/bn0_conv_0_bn_project/beta:0']
shift['/bn0_conv_0_bn_project/bn0_conv_0_bn_project/beta:0'] = npShift

npShift = np.loadtxt('other/shiftbn0_conv_0_bn_project_gamma.txt')
del shift['/bn0_conv_0_bn_project/bn0_conv_0_bn_project/gamma:0']
shift['/bn0_conv_0_bn_project/bn0_conv_0_bn_project/gamma:0'] = npShift

npShift = np.loadtxt('other/shiftbn0_conv_0_bn_project_mean.txt')
del shift['/bn0_conv_0_bn_project/bn0_conv_0_bn_project/moving_mean:0']
shift['/bn0_conv_0_bn_project/bn0_conv_0_bn_project/moving_mean:0'] = npShift

npShift = np.loadtxt('other/shiftbn0_conv_0_bn_project_variance.txt')
del shift['/bn0_conv_0_bn_project/bn0_conv_0_bn_project/moving_variance:0']
shift['/bn0_conv_0_bn_project/bn0_conv_0_bn_project/moving_variance:0'] = npShift

#######################################################

for i in range(16):
	I = i+1
	idx = str(I)

	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_beta_depth.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/beta:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/beta:0'] = npShift
	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_beta_expand.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/beta:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/beta:0'] = npShift

	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_beta_project.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/beta:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/beta:0'] = npShift
	
	##############################################
	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_gam_depth.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/gamma:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/gamma:0'] = npShift
	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_gam_expand.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/gamma:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/gamma:0'] = npShift

	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_gam_project.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/gamma:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/gamma:0'] = npShift
	
	##############################################
	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_mean_depth.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/moving_mean:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/moving_mean:0'] = npShift
	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_mean_expand.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/moving_mean:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/moving_mean:0'] = npShift

	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_mean_project.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/moving_mean:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/moving_mean:0'] = npShift
	
	##############################################
	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_var_depth.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/moving_variance:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/moving_variance:0'] = npShift
	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_var_expand.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/moving_variance:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/moving_variance:0'] = npShift

	npShift = np.loadtxt("bn"+idx+"/shiftnpbn_var_project.txt")
	del shift['/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/moving_variance:0']
	shift['/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/moving_variance:0'] = npShift

##################################################################################

npShift = np.loadtxt("other/shiftConv1_beta.txt")
del shift['/bn_Conv1/bn_Conv1/beta:0']
shift['/bn_Conv1/bn_Conv1/beta:0'] = npShift

npShift = np.loadtxt("other/shiftConv1_gamma.txt")
del shift['/bn_Conv1/bn_Conv1/gamma:0']
shift['/bn_Conv1/bn_Conv1/gamma:0'] = npShift

npShift = np.loadtxt("other/shiftConv1_mean.txt")
del shift['/bn_Conv1/bn_Conv1/moving_mean:0']
shift['/bn_Conv1/bn_Conv1/moving_mean:0'] = npShift

npShift = np.loadtxt("other/shiftConv1_variance.txt")
del shift['/bn_Conv1/bn_Conv1/moving_variance:0']
shift['/bn_Conv1/bn_Conv1/moving_variance:0'] = npShift

##################################################################################

npShift = np.loadtxt('other/shiftmobl0_depth.txt')
OGshape = hdf.get('/mobl0_conv_0_depthwise/mobl0_conv_0_depthwise/depthwise_kernel:0')
newNpy = npShift.reshape(OGshape.shape)
del shift['/mobl0_conv_0_depthwise/mobl0_conv_0_depthwise/depthwise_kernel:0']
shift['/mobl0_conv_0_depthwise/mobl0_conv_0_depthwise/depthwise_kernel:0'] = newNpy


npShift = np.loadtxt('other/shiftmobl0_project.txt')
OGshape = hdf.get('/mobl0_conv_0_project/mobl0_conv_0_project/kernel:0')
newNpy = npShift.reshape(OGshape.shape)
del shift['/mobl0_conv_0_project/mobl0_conv_0_project/kernel:0']
shift['/mobl0_conv_0_project/mobl0_conv_0_project/kernel:0'] = newNpy

##################################################################################

for j in range(16):
	I = j+1
	idx = str(I)

	npShift = np.loadtxt("mobl"+idx+"/shiftmobl"+idx+"depth.txt")
	OGshape = hdf.get('/mobl'+idx+'_conv_'+idx+'_depthwise/mobl'+idx+'_conv_'+idx+'_depthwise/depthwise_kernel:0')
	newNpy = npShift.reshape(OGshape.shape)
	del shift['/mobl'+idx+'_conv_'+idx+'_depthwise/mobl'+idx+'_conv_'+idx+'_depthwise/depthwise_kernel:0']
	shift['/mobl'+idx+'_conv_'+idx+'_depthwise/mobl'+idx+'_conv_'+idx+'_depthwise/depthwise_kernel:0'] = newNpy

	##############################################
	npShift = np.loadtxt("mobl"+idx+"/shiftmobl"+idx+"expand.txt")
	OGshape = hdf.get('/mobl'+idx+'_conv_'+idx+'_expand/mobl'+idx+'_conv_'+idx+'_expand/kernel:0')
	newNpy = npShift.reshape(OGshape.shape)
	del shift['/mobl'+idx+'_conv_'+idx+'_expand/mobl'+idx+'_conv_'+idx+'_expand/kernel:0']
	shift['/mobl'+idx+'_conv_'+idx+'_expand/mobl'+idx+'_conv_'+idx+'_expand/kernel:0'] = newNpy

	##############################################
	npShift = np.loadtxt("mobl"+idx+"/shiftmobl"+idx+"project.txt")
	OGshape = hdf.get('/mobl'+idx+'_conv_'+idx+'_project/mobl'+idx+'_conv_'+idx+'_project/kernel:0')
	newNpy = npShift.reshape(OGshape.shape)
	del shift['/mobl'+idx+'_conv_'+idx+'_project/mobl'+idx+'_conv_'+idx+'_project/kernel:0']
	shift['/mobl'+idx+'_conv_'+idx+'_project/mobl'+idx+'_conv_'+idx+'_project/kernel:0'] = newNpy













