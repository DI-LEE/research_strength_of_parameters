import numpy as np
import pandas as pd
import h5py

hdf = h5py.File('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5','r')

Conv1_kernel = hdf.get('/Conv1/Conv1/kernel:0')
npConv1_kernel = np.array(Conv1_kernel)
FlatConv1_kernel = npConv1_kernel.flatten()
np.savetxt("other/Conv1.txt", FlatConv1_kernel)
#4dim Conv1/Conv1/kernel:0 to 1dim txt file 1

Conv_1_kernel = hdf.get('/Conv_1/Conv_1/kernel:0')
npConv_1_kernel = np.array(Conv_1_kernel)
FlatConv_1_kernel = npConv_1_kernel.flatten()
np.savetxt("other/Conv_1.txt", FlatConv_1_kernel)
#4dim Conv_1/Conv_1/kernel:0 to 1dim txt file 2  !!!!!

Conv_1_beta = hdf.get('/Conv_1_bn/Conv_1_bn/beta:0')
npConv_1_beta = np.array(Conv_1_beta)
np.savetxt("other/Conv_1_bn_beta.txt", npConv_1_beta)
#save Conv_1_bn/Conv_1_bn/beta:0 txt file 3

Conv_1_mean = hdf.get('/Conv_1_bn/Conv_1_bn/moving_mean:0')
npConv_1_mean = np.array(Conv_1_mean)
np.savetxt("other/Conv_1_bn_mean.txt", npConv_1_mean)
#save Conv_1_bn/Conv_1_bn/moving_maen:0 txt file 4

Conv_1_var = hdf.get('/Conv_1_bn/Conv_1_bn/moving_variance:0')
npConv_1_var = np.array(Conv_1_var)
np.savetxt("other/Conv_1_bn_var.txt", npConv_1_var)
#save Conv_1_bn/Conv_1_bn/moving_variance:0 txt file 5

Conv_1_gam = hdf.get('/Conv_1_bn/Conv_1_bn/gamma:0')
npConv_1_gam = np.array(Conv_1_gam)
np.savetxt("other/Conv_1_bn_gam.txt", npConv_1_gam)
#save Conv_1_bn/Conv_1_bn/gamma:0 txt file 6

##################################################################################

Logits_bias = hdf.get('/Logits/Logits/bias:0')
npLogits_bias = np.array(Logits_bias)
np.savetxt("other/Logits_bias.txt", npLogits_bias)
#save Logits/Logits/bias:0 txt file 7

Logits_kernel = hdf.get('/Logits/Logits/kernel:0')
npLogits_kernel = np.array(Logits_kernel)
FlatLogits_kernel = npLogits_kernel.flatten()
np.savetxt("other/Logits_kernel.txt", FlatLogits_kernel)
#save Logits/Logits/beta:0 txt file 8

################################################################################## !!!!!!

bn0_conv_0_beta = hdf.get('/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/beta:0')
npbn0_conv_0_beta = np.array(bn0_conv_0_beta)
np.savetxt("other/bn0_conv_0_bn_depthwise_beta.txt", npbn0_conv_0_beta)
#save bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/beta:0 txt file 9

bn0_conv_0_gamma = hdf.get('/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/gamma:0')
npbn0_conv_0_gamma = np.array(bn0_conv_0_gamma)
np.savetxt("other/bn0_conv_0_bn_depthwise_gamma.txt", npbn0_conv_0_gamma)
#save bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/gamma:0 txt file 10

bn0_conv_0_mean = hdf.get('/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/moving_mean:0')
npbn0_conv_0_mean = np.array(bn0_conv_0_mean)
np.savetxt("other/bn0_conv_0_bn_depthwise_mean.txt", npbn0_conv_0_mean)
#save bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/moving_mean:0 txt file 11

bn0_conv_0_variance = hdf.get('/bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/moving_variance:0')
npbn0_conv_0_variance = np.array(bn0_conv_0_variance)
np.savetxt("other/bn0_conv_0_bn_depthwise_variance.txt", npbn0_conv_0_variance)
#save bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/moving_variance:0 txt file 12

##################################################################################   !!!!!!!

bn0_conv_0_pj_beta = hdf.get('/bn0_conv_0_bn_project/bn0_conv_0_bn_project/beta:0')
npbn0_conv_0_pj_beta = np.array(bn0_conv_0_pj_beta)
np.savetxt("other/bn0_conv_0_bn_project_beta.txt", npbn0_conv_0_pj_beta)
#save bn0_conv_0_bn_project/bn0_conv_0_bn_project/beta:0 txt file 13

bn0_conv_0_pj_gamma = hdf.get('/bn0_conv_0_bn_project/bn0_conv_0_bn_project/gamma:0')
npbn0_conv_0_pj_gamma = np.array(bn0_conv_0_pj_gamma)
np.savetxt("other/bn0_conv_0_bn_project_gamma.txt", npbn0_conv_0_pj_gamma)
#save bn0_conv_0_bn_project/bn0_conv_0_bn_project/gamma:0 txt file 14

bn0_conv_0_pj_mean = hdf.get('/bn0_conv_0_bn_project/bn0_conv_0_bn_project/moving_mean:0')
npbn0_conv_0_pj_mean = np.array(bn0_conv_0_pj_mean)
np.savetxt("other/bn0_conv_0_bn_project_mean.txt", npbn0_conv_0_pj_mean)
#save bn0_conv_0_bn_project/bn0_conv_0_bn_project/moving_mean:0 txt file 15

bn0_conv_0_pj_variance = hdf.get('/bn0_conv_0_bn_project/bn0_conv_0_bn_project/moving_variance:0')
npbn0_conv_0_pj_variance = np.array(bn0_conv_0_pj_variance)
np.savetxt("other/bn0_conv_0_bn_project_variance.txt", npbn0_conv_0_pj_variance)
#save bn0_conv_0_bn_depthwise/bn0_conv_0_bn_depthwise/moving_variance:0 txt file 16

##################################################################################

for i in range(16):
	I = i+1
	idx = str(I)
	bn_beta_depth = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/beta:0')
	npbn_beta_depth = np.array(bn_beta_depth)
	np.savetxt("bn"+idx+"/npbn_beta_depth.txt", npbn_beta_depth)
	
	bn_beta_expand = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/beta:0')
	npbn_beta_expand = np.array(bn_beta_expand)
	np.savetxt("bn"+idx+"/npbn_beta_expand.txt", npbn_beta_expand)

	bn_beta_project = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/beta:0')
	npbn_beta_project = np.array(bn_beta_project)
	np.savetxt("bn"+idx+"/npbn_beta_project.txt", npbn_beta_project)
	##############################################
	bn_gam_depth = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/gamma:0')
	npbn_gam_depth = np.array(bn_gam_depth)
	np.savetxt("bn"+idx+"/npbn_gam_depth.txt", npbn_gam_depth)

	bn_gam_expand = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/gamma:0')
	npbn_gam_expand = np.array(bn_gam_expand)
	np.savetxt("bn"+idx+"/npbn_gam_expand.txt", npbn_gam_expand)

	bn_gam_project = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/gamma:0')
	npbn_gam_project = np.array(bn_gam_project)
	np.savetxt("bn"+idx+"/npbn_gam_project.txt", npbn_gam_project)
	##############################################
	bn_mean_depth = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/moving_mean:0')
	npbn_mean_depth = np.array(bn_mean_depth)
	np.savetxt("bn"+idx+"/npbn_mean_depth.txt", npbn_mean_depth)

	bn_mean_expand = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/moving_mean:0')
	npbn_mean_expand = np.array(bn_mean_expand)
	np.savetxt("bn"+idx+"/npbn_mean_expand.txt", npbn_mean_expand)

	bn_mean_project = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/moving_mean:0')
	npbn_mean_project = np.array(bn_mean_project)
	np.savetxt("bn"+idx+"/npbn_mean_project.txt", npbn_mean_project)
	##############################################
	bn_var_depth = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_depthwise/bn'+idx+'_conv_'+idx+'_bn_depthwise/moving_variance:0')
	npbn_var_depth = np.array(bn_var_depth)
	np.savetxt("bn"+idx+"/npbn_var_depth.txt", npbn_var_depth)

	bn_var_expand = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_expand/bn'+idx+'_conv_'+idx+'_bn_expand/moving_variance:0')
	npbn_var_expand = np.array(bn_var_expand)
	np.savetxt("bn"+idx+"/npbn_var_expand.txt", npbn_var_expand)

	bn_var_project = hdf.get('/bn'+idx+'_conv_'+idx+'_bn_project/bn'+idx+'_conv_'+idx+'_bn_project/moving_variance:0')
	npbn_var_project = np.array(bn_var_project)
	np.savetxt("bn"+idx+"/npbn_var_project.txt", npbn_var_project)

##################################################################################
Conv1_beta = hdf.get('/bn_Conv1/bn_Conv1/beta:0')
npConv1_beta = np.array(Conv1_beta)
np.savetxt("other/Conv1_beta.txt", npConv1_beta)
#save bn_Conv1/bn_Conv1/beta:0 txt file 17

Conv1_gamma = hdf.get('/bn_Conv1/bn_Conv1/gamma:0')
npConv1_gamma = np.array(Conv1_gamma)
np.savetxt("other/Conv1_gamma.txt", npConv1_gamma)
#save bn_Conv1/bn_Conv1/gamma:0 txt file 18

Conv1_mean = hdf.get('/bn_Conv1/bn_Conv1/moving_mean:0')
npConv1_mean = np.array(Conv1_mean)
np.savetxt("other/Conv1_mean.txt", npConv1_mean)
#save '/bn_Conv1/bn_Conv1/moving_mean:0' txt file  19

Conv1_variance = hdf.get('/bn_Conv1/bn_Conv1/moving_variance:0')
npConv1_variance = np.array(Conv1_variance)
np.savetxt("other/Conv1_variance.txt", npConv1_variance)
#save bn_Conv1/bn_Conv1/moving_variance:0 txt file 20

##################################################################################

depth_kernel = hdf.get('/mobl0_conv_0_depthwise/mobl0_conv_0_depthwise/depthwise_kernel:0')
npdepth_kernel = np.array(depth_kernel)
Flatdepth_kernel = npdepth_kernel.flatten()
np.savetxt("other/mobl0_depth.txt", Flatdepth_kernel)
#4dim mobl0_conv_0_depthwise/mobl0_conv_0_depthwise/depthwise_kernel:0 to 1dim txt file 21

project_kernel = hdf.get('/mobl0_conv_0_project/mobl0_conv_0_project/kernel:0')
npproject_kernel = np.array(project_kernel)
Flatproject_kernel = npproject_kernel.flatten()
np.savetxt("other/mobl0_project.txt", Flatproject_kernel)
#4dim mobl0_conv_0_depthwise/mobl0_conv_0_depthwise/depthwise_kernel:0 to 1dim txt file 22

##################################################################################

for j in range(16):
	I = j+1
	idx = str(I)
	depth_kernel = hdf.get('/mobl'+idx+'_conv_'+idx+'_depthwise/mobl'+idx+'_conv_'+idx+'_depthwise/depthwise_kernel:0')
	npdepth_kernel = np.array(depth_kernel)
	Flatdepth_kernel = npdepth_kernel.flatten()
	np.savetxt("mobl"+idx+"/mobl"+idx+"depth.txt", Flatdepth_kernel)
	
	##############################################
	expand_kernel = hdf.get('/mobl'+idx+'_conv_'+idx+'_expand/mobl'+idx+'_conv_'+idx+'_expand/kernel:0')
	npexpand_kernel = np.array(expand_kernel)
	Flatexpand_kernel = npexpand_kernel.flatten()
	np.savetxt("mobl"+idx+"/mobl"+idx+"expand.txt", Flatexpand_kernel)

	##############################################
	project_kernel = hdf.get('/mobl'+idx+'_conv_'+idx+'_project/mobl'+idx+'_conv_'+idx+'_project/kernel:0')
	npproject_kernel = np.array(project_kernel)
	Flatproject_kernel = npproject_kernel.flatten()
	np.savetxt("mobl"+idx+"/mobl"+idx+"project.txt", Flatproject_kernel)


