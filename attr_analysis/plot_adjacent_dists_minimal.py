import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

import logomaker


def plot_heatmap_in_axs(vals_mat, ax, 
		is_pre=False, desc=None, cmap='PRGn', max_attr=None, min_attr=None,
		plot_cbar=False, x_ticks=[]):
	"""Plot heatmap in given axs.
	
	Args:
		vals_mat: numpy array to plot as heatmap
		ax: matplotlib axes to plot heatmap in
		is_pre: whether this is a pre- or post- STR sequence. Changes 
			numbering to counting down (e.g. x labels [-3, -2, -1] when True
			[1, 2, 3] when False (default)).
		cmap: matplotlib colormap to use
		max_attr: maximum value to plot on heatmap, used for scaling colors
		min_attr: minimum value to plot on heatmap, used for scaling colors
		plot_cbar: whether to plot a colorbar
		x_ticks: list of x tick labels to use, defaults to [] (no ticks)
	"""
	if max_attr is None:
		max_attr = vals_mat.max()
	if min_attr is None:
		min_attr = vals_mat.min()

	sns.heatmap(
		data=vals_mat,
		vmin=min_attr,
		vmax=max_attr,
		center=0,
		yticklabels=['A', 'C', 'G', 'T'],
		xticklabels=x_ticks,
		ax=ax,
		cmap=cmap,
		cbar=plot_cbar
	)

	return ax


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


def reformat_attribution_pkl(attr_data):
	"""Reformat attribution data from as read from pkl to numpy arrays
	or pandas dataframes.
	"""
	attr_data['pre'] = np.stack(attr_data['pre'])
	attr_data['post'] = np.stack(attr_data['post'])
	attr_data['pre_strings'] = np.stack(attr_data['pre_strings'])
	attr_data['post_strings'] = np.stack(attr_data['post_strings'])

	# fix a typo
	if 'predicitons' in attr_data.keys():
		attr_data['predictions'] = attr_data.pop('predicitons')

	attr_data['predictions'] = np.array(attr_data['predictions'])
	attr_data['labels'] = np.array(attr_data['labels'])
	attr_data['sample_data'] = pd.DataFrame(attr_data['sample_data'])

	return attr_data


def add_prediction_features(attr_data, cutoff=.5):
	"""Add binary prediction features and their correctness to attr_data."""
	attr_data['binary_pred'] = (attr_data['predictions'] > cutoff).astype(int)
	attr_data['correct_pred'] = (attr_data['binary_pred'] == attr_data['labels'])
	attr_data['true_pos'] = (attr_data['binary_pred'] == 1) & (attr_data['labels'] == 1)
	attr_data['true_neg'] = (attr_data['binary_pred'] == 0) & (attr_data['labels'] == 0)
	return attr_data


def subset_attr_data(attr_data, subset_mask):
	attr_data = copy.deepcopy(attr_data)

	for k,v in attr_data.items():
		if isinstance(v, np.ndarray):
			attr_data[k] = v[subset_mask]
		elif isinstance(v, pd.DataFrame):
			attr_data[k] = v.loc[subset_mask]

	return attr_data


def get_phi_corrs(X, y):
	"""Get phi correlation for every position in batch of binary 2D 
	arrays X (e.i. a 3D array) with y.
	"""
	X_mats_flat = X.reshape(X.shape[0], -1)

	# get correlation for each position
	phi_corrs = []
	for i in range(X_mats_flat.shape[1]):
		phi_corrs.append(metrics.matthews_corrcoef(y, X_mats_flat[:, i]))

	return np.array(phi_corrs).reshape(X.shape[1:])


def get_base_probs(subset_strings):
	count_dict = {
		k: v for k,v in zip(*np.unique(subset_strings, return_counts=True))
	}
	counts = np.array([
		count_dict['A'], count_dict['C'], count_dict['G'], count_dict['T']
	])
	return counts / counts.sum()


def get_subset_plot_matrices(subset_data, pattern, n_per_side, str_pad_size,
		pre_seq=True, base_probs=None, return_base_probs=False, name=None):
	"""Generate matrices used to make plots about a set of sequences.
	
	Args:
		subset_data: dict of the normal type for attribution data
		pattern: pattern that STR starts/ends with
		n_per_side: number of loci to use
		str_pad_size: number of initial loci that are STR pattern padding
		pre_seq: whether subset_data contains pre- or post- STR sequences.
			Default is True (pre-STR).
		base_probs: optional array of base probabilities to use instead of
			generating from all samples in subset_data using get_base_probs.
		return_base_probs: whether to return base probabilities used
		name: optional name to include in returned dict
	
	Returns:
		(subset_res, base_probs) if return_base_probs is True, else subset_res

		subset_res is a dict with keys: 'phi_corrs', 'mean_attr', 'attr_mgm', 
			'ppm', 'pwm', and optionally 'name'
	"""
	subset_res = dict()
	if name is not None:
		subset_res['name'] = name

	subset_res['count'] = len(subset_data['sample_data'])

	# get needed features
	attrs = subset_data['pre'][:,:4:,-n_per_side-str_pad_size:-str_pad_size]
	subset_strings = np.array([
		np.array(list(c[-n_per_side-str_pad_size:-str_pad_size])) for 
			c in subset_data['pre_strings']
	])

	# get phi correlations
	bin_seqs = np.stack([
		(subset_strings == 'A').astype(int),
		(subset_strings == 'C').astype(int),
		(subset_strings == 'G').astype(int),
		(subset_strings == 'T').astype(int)
	]).transpose(1,0,2)
	phi_corrs = get_phi_corrs(bin_seqs, subset_data['labels'])
	subset_res['phi_corrs'] = phi_corrs

	# get mean and median attribution scores
	if mean_ignore_0:
		attrs_nan = attrs.copy()
		attrs_nan[attrs_nan == 0] = np.nan
		with np.errstate(all='ignore'):
			mean_attr = np.nan_to_num(np.nanmean(attrs_nan, 0))
	else:
		mean_attr = attrs.mean(0)
	subset_res['mean_attr'] = mean_attr

	if mean_ignore_0:
		attrs_nan = attrs.copy()
		attrs_nan[attrs_nan == 0] = np.nan
		with np.errstate(all='ignore'):
			med_attr = np.nan_to_num(np.nanmedian(attrs_nan, 0))
	else:
		med_attr = attrs.median(0)
	subset_res['med_attr'] = med_attr

	# get mean greatest magnitude by position
	attr_mgm = np.max(np.abs(attrs), axis=1).mean(0)
	subset_res['attr_mgm'] = attr_mgm

	# get strings for next 3 ppm based feats
	counts_mat = np.stack([
		(subset_strings == 'A').sum(0),
		(subset_strings == 'C').sum(0),
		(subset_strings == 'G').sum(0),
		(subset_strings == 'T').sum(0)
	])
	ppm = counts_mat / counts_mat.sum(0, keepdims=True)
	ppm_df = pd.DataFrame(ppm.T, columns=['A', 'C', 'G', 'T'])
	subset_res['ppm'] = ppm_df

	# ppm w/ psuedocounts to position weight matrix PWM (log-likelihoods)
	#	wrt subset
	if base_probs is None:
		base_probs = get_base_probs(subset_strings)
	counts_mat_pc = counts_mat + 1
	ppm = counts_mat_pc / counts_mat_pc.sum(0, keepdims=True)
	if len(base_probs.shape) == 1:
		pwm = np.log2(ppm / base_probs[:, None])
	else:
		with np.errstate(divide='ignore'):
			pwm = np.nan_to_num(np.log2(ppm / base_probs))

	# redo last row knowing that value cannot be last base in motif
	if pre_seq:
		valid_base_mask = np.array(['A', 'C', 'G', 'T']) != pattern[-1]
		pwm[:, -1][~valid_base_mask] = 0.0
		if len(base_probs.shape) == 1:
			pwm[:, -1][valid_base_mask] = (
				pwm[:, -1][valid_base_mask] / np.sum(pwm[:, -1][valid_base_mask])
			)
	elif (not pre_seq) and len(base_probs.shape) == 1:
		raise NotImplementedError()
	pwm_df = pd.DataFrame(pwm.T, columns=['A', 'C', 'G', 'T'])
	subset_res['pwm'] = pwm_df

	if return_base_probs:
		return subset_res, base_probs
	else:
		return subset_res


if __name__ == '__main__':
	# Options
	attrs_dir = '../prediction/training_output'
	label_version = [
		'v1-mfr0_005_mnc2000-m6_5',
		'v1-mfr0_0025_mnc2000-m7_5'
	][0]
	model_version = 'tscc_version_0'
	attr_file = 'ig_global_val_test.pkl'

	save_fig = True
	show_fig = False
	flip_attr_plot_so_pos_label_up = False

	str_motif_len = 2
	str_pad_size = 4

	n_per_side = 24

	# Load single attr type
	attr_data = pd.read_pickle(os.path.join(
		attrs_dir, label_version, model_version, 'attributions', attr_file
	))

	# Reformat data and add prediction features
	attr_data = reformat_attribution_pkl(attr_data)
	attr_data = add_prediction_features(attr_data)
	attr_data['sample_data']['alpha_motif'] = attr_data['sample_data'].motif.apply(
		lambda x: ''.join(sorted([*x])) # motif name with letters sorted alphabetically
	)

	# tp_data = subset_attr_data(
	# 	attr_data, 
	# 	(attr_data['binary_pred'] == 1) & (attr_data['labels'] == 1)
	# )
	# tn_data = subset_attr_data(
	# 	attr_data, 
	# 	(attr_data['binary_pred'] == 0) & (attr_data['labels'] == 0)
	# )

	# Add data on start/end of STR, since ploting will be done 
	#	independantly for each type
	min_count = 0 # must be at least this many examples of STR start/end type

	STR_starts = np.array([
		s[-str_pad_size : -str_pad_size+str_motif_len] for s in attr_data['pre_strings']
	])
	STR_ends = np.array([
		s[str_pad_size-str_motif_len : str_pad_size] for s in attr_data['post_strings']
	])

	print(*zip(*np.unique(STR_starts, return_counts=True)))
	print(*zip(*np.unique(STR_ends, return_counts=True)))

	starts_to_plot = [
		s for s,c in zip(*np.unique(STR_starts, return_counts=True)) if c >= min_count
	]
	ends_to_plot = [
		s for s,c in zip(*np.unique(STR_ends, return_counts=True)) if c >= min_count
	]

	all_strings = np.array([
		np.array(list(c[-n_per_side-str_pad_size:-str_pad_size])) for 
			c in attr_data['pre_strings']
	])

	# For each STR start pattern subset, cluster seqs by attribution weights
	'''if mean_ignore_0 true, mean attribution for a (base,position) will 
	be caluculated only for seqs that contain that base at that position. 
	This in effect ignores the 0s added by using a global atttribution.
	'''
	mean_ignore_0 = True
	all_res = dict()			

	for start_pattern in tqdm(starts_to_plot, desc='data preprocessing'):
		# subset data
		subset_data = subset_attr_data(
			attr_data,
			STR_starts == start_pattern
		)
		
		plot_title = 'Pre-{} (n = {})'.format(
			start_pattern, len(subset_data['labels'])
		)
		print(plot_title)
		all_res[plot_title] = []

		res, base_probs = get_subset_plot_matrices(
			subset_data,
			pattern=start_pattern,
			n_per_side=n_per_side,
			str_pad_size=str_pad_size,
			pre_seq=True,
			return_base_probs=True,
			name='All {}'.format(plot_title),
		)
		all_res[plot_title].append(res)

		# het samples
		het_data = subset_attr_data(subset_data, subset_data['labels'] == 1)
		# het_res = get_subset_plot_matrices(
		# 	het_data,
		# 	pattern=start_pattern,
		# 	n_per_side=n_per_side,
		# 	str_pad_size=str_pad_size,
		# 	pre_seq=True,
		# 	name='Het {}'.format(plot_title),
		# 	base_probs=all_res[plot_title][0]['ppm'].values.T
		# )

		# TP samples
		tp_data = subset_attr_data(het_data, het_data['binary_pred'] == 1)
		tp_res = get_subset_plot_matrices(
			het_data,
			pattern=start_pattern,
			n_per_side=n_per_side,
			str_pad_size=str_pad_size,
			pre_seq=True,
			name='TP Het {}'.format(plot_title),
			base_probs=all_res[plot_title][0]['ppm'].values.T
		)

		# non-het samples
		homo_data = subset_attr_data(subset_data, subset_data['labels'] == 0)
		# homo_res = get_subset_plot_matrices(
		# 	homo_data,
		# 	pattern=start_pattern,
		# 	n_per_side=n_per_side,
		# 	str_pad_size=str_pad_size,
		# 	pre_seq=True,
		# 	name='non-Het {}'.format(plot_title),
		# 	base_probs=all_res[plot_title][0]['ppm'].values.T
		# )

		# TN samples
		tn_data = subset_attr_data(homo_data, homo_data['binary_pred'] == 0)
		tn_res = get_subset_plot_matrices(
			tn_data,
			pattern=start_pattern,
			n_per_side=n_per_side,
			str_pad_size=str_pad_size,
			pre_seq=True,
			name='TN non-Het {}'.format(plot_title),
			base_probs=all_res[plot_title][0]['ppm'].values.T
		)

		# add additional plots for subsubsets that do PWM vs opposite label all
		# valid_base_mask = np.array(['A', 'C', 'G', 'T']) != start_pattern[-1]

		with np.errstate(divide='ignore'):
			tp_res['pwm_oppo'] = pd.DataFrame(
				np.nan_to_num(np.log2(tp_res['ppm'] / tn_res['ppm'])),
				columns=['A', 'C', 'G', 'T']
			)

			tn_res['pwm_oppo'] = pd.DataFrame(
				np.nan_to_num(np.log2(tn_res['ppm'] / tp_res['ppm'])),
				columns=['A', 'C', 'G', 'T']
			)

		# add subsubset data
		all_res[plot_title].append(tp_res)
		all_res[plot_title].append(tn_res)

	# For sanity check
	print([(d[0]['name'],d[0]['count']) for _,d in all_res.items()])
	(attr_data['sample_data'].alpha_motif == 'AC').sum()
	(attr_data['sample_data'].alpha_motif == 'GT').sum()

	# Plot 
	for pattern,col_data in (pbar := 
			tqdm(all_res.items(), desc='Plotting', total=len(all_res.items()))):
		pbar.set_description(f"Plotting {pattern}")
		fig, axs = plt.subplots(3, sharex=True)#, figsize=(20, 8))
		fig.suptitle(f'{pattern}')

		max_attr_mag = 0
		min_attr_mag = 0

		rel_pwm_abs_max = 0
		max_mgm = 0

		# importance by posititon
		sns.scatterplot(
			x=np.array(list(range(col_data[0]['attr_mgm'].shape[0]))) + 0.,
			y=col_data[0]['attr_mgm'],
			ax=axs[0],
			vmin=0
		)
		axs[0].set_title('Median Attribution Score by Relative Position')
		axs[0].set_ylim(bottom=0)

		# differnce in attrs
		logomaker.Logo(
			pd.DataFrame(
				(col_data[1]['med_attr'] - col_data[2]['med_attr']).T,
				columns=['A', 'C', 'G', 'T']
			),
			ax=axs[1],
			center_values=True,
		)
		axs[1].set_title('Mean Centered Difference in Median Attributions (TP - TN)')

		# relative log odds
		logomaker.Logo(col_data[1]['pwm_oppo'], ax=axs[2])#, center_values=True)
		axs[2].set_title('PPM Log Odds (TP / TN)')
		y_abs_max = np.abs(axs[2].dataLim.get_points()[((0,1), (1,1))]).max()
		axs[2].set_ylim(-y_abs_max, y_abs_max)

		# save and/or show figure
		tick_vals = np.array(list(range(n_per_side, -1, -4)))
		plt.setp(
			axs, 
			xticks=tick_vals, 
			xticklabels=np.abs(tick_vals - 32) * -1 - 1
		)

		plt.tight_layout(h_pad=0.35)

		if save_fig:
			plot_save_dir = os.path.join(
				'plots_min_adj_seq_centattr', 
				model_version,
				# pattern.replace(' ', '')
			)
			if not os.path.exists(plot_save_dir):
				os.makedirs(plot_save_dir)

			plot_fname = '{}_{}_all.png'.format(pattern, n_per_side)
			plot_save_file = os.path.join(plot_save_dir, plot_fname)
			plt.savefig(plot_save_file)
		if show_fig:
			plt.show()

		plt.close()